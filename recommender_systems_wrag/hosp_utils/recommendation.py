import openai
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime, timedelta
from utils.feedback_manager import FeedbackManager

config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class HospitalRecommender:
    def __init__(self, model_name='text-embedding-3-small', db_connection=None):
        self.model_name = model_name
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.policy_network = PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        self.gamma = 0.99
        self.K = 15
        self.feedback_manager = FeedbackManager(db_connection) if db_connection else None

    def get_embedding(self, text_list, batch_size=256):
        """
        OpenAI API를 이용하여 텍스트 임베딩 생성
        :param text_list: 리스트 형식의 입력 텍스트 데이터
        :param batch_size: batch 단위로 처리하게끔 지정
        :return: numpy array 형태의 임베딩 결과
        """
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            try:
                response = openai.embeddings.create(
                    model=self.model_name,  #최신 모델 사용 추천
                    input=batch,
                    encoding_format="float"
                )
                
                embeddings.extend([embedding.embedding for embedding in response.data])

            except openai.OpenAIError as e:
                #요청 실패 시, 빈 벡터 반환(1536은 text-embedding-ada-002 모델의 기본 차원)
                zero_vector = np.zeros((len(batch), 1536))
                embeddings.extend(zero_vector)

        return np.array(embeddings)

    def embed_user_profile(self, basic_info, health_info, suspected_disease=None, department=None):
        """
        사용자 건강 정보 및 기초 정보를 OpenAI Embedding을 활용하여 벡터화
        """

        text_data = "의심질병: {suspected_disease}, 진료과: {department}, 가족력: {family}, 성별: {gender}, 병력: {past}, 복용약: {med}".format(
            gender=basic_info.get("gender", "unknown"),
            past=health_info.get("pastHistory", "unknown"),
            family=health_info.get("familyHistory", "unknown"),
            med=health_info.get("nowMedicine", "unknown"),
            suspected_disease=suspected_disease or "unknown",
            department=department or "unknown"
        )

        text_embedding = self.get_embedding([text_data])[0] 

        return text_embedding

    def embed_hospital_data(self, hospitals_df):
        """
        병원 데이터를 OpenAI API 임베딩으로 변환
        """
        for col in hospitals_df.columns:
            if hospitals_df[col].dtype == "object":
                hospitals_df[col] = hospitals_df[col].fillna("unknown").replace("", "unknown")
            elif hospitals_df[col].dtype in ["float64", "int64"]:
                hospitals_df[col] = hospitals_df[col].fillna(0)
        
        #병원 데이터를 하나의 문장으로 결합하여 OpenAI API로 벡터화
        hospital_sentences = hospitals_df.apply(
            lambda row: f"병원명: {row['name']}, 병원유형: {row['clcdnm']}, 진료과: {row['department']}",
           axis=1
        ).tolist()

        #3. API 호출 최적화(중복된 병원명 제거→중복 API 호출 방지)
        unique_sentences = list(set(hospital_sentences))  #중복 제거
        unique_embeddings = self.get_embedding(unique_sentences, batch_size=256)
        print(unique_embeddings.shape)
        #4️.병원명 기준으로 임베딩 매핑
        embedding_dict = dict(zip(unique_sentences, unique_embeddings))
        hospital_embeddings = np.array([embedding_dict[sentence] for sentence in hospital_sentences])

        return hospital_embeddings

    def calculate_base_score(self, hospital_df):
        """
        기본 점수 계산 (위치 기반 필터링)
        - 이동 시간을 기반으로 한 거리 기반 점수
        """
        # 이동 시간을 초 단위로 변환
        total_travel_time = (
            hospital_df['transit_travel_time_h'] * 3600 +
            hospital_df['transit_travel_time_m'] * 60 +
            hospital_df['transit_travel_time_s']
        )
        
        # 이동 시간이 짧을수록 높은 점수
        base_scores = 1 / (1 + total_travel_time)
        return base_scores

    def calculate_reward(self, hospital_data, user_embedding, hospital_embedding, member_id=None):
        """
        병원 데이터와 사용자 임베딩을 기반으로 보상 계산
        - 기본 점수(위치 기반)와 콘텐츠 기반 점수를 결합한 하이브리드 보상 함수
        - 장기적 보상에 gamma 적용
        - 보상 정규화 적용
        """
        # 1. 기본 점수 (위치 기반 필터링) - 40% 가중치
        base_score = self.calculate_base_score(pd.DataFrame([hospital_data])).iloc[0]
        
        # 2. 콘텐츠 기반 점수 (임베딩 유사도) - 60% 가중치
        similarity = cosine_similarity(user_embedding.reshape(1, -1), 
                                    hospital_embedding.reshape(1, -1))[0][0]
        # 코사인 유사도를 0~1 범위로 정규화
        similarity = (similarity + 1) / 2  # -1~1 -> 0~1
        
        # 3. 즉각적인 보상 계산 (0~1 범위)
        immediate_reward = 0.4 * base_score + 0.6 * similarity
        
        # 4. 장기적 보상 계산 (이전 선택 기록 기반)
        if member_id and self.feedback_manager:
            feedback_data = self.feedback_manager.get_hospital_feedback(member_id)
            if not feedback_data.empty:
                long_term_reward = self.feedback_manager.calculate_hospital_reward(hospital_data, feedback_data)
                # 장기적 보상을 0~1 범위로 정규화
                long_term_reward = min(long_term_reward, 1.0)
                # gamma를 사용하여 장기적 보상 할인
                return immediate_reward + self.gamma * long_term_reward
        
        return immediate_reward

    def recommend_hospitals(self, user_embedding, hospital_embeddings, hospitals_df, member_id=None):
        """
        하이브리드 추천 시스템 (REINFORCE + 콘텐츠 기반 필터링)
        - REINFORCE 알고리즘을 통한 정책 학습
        - 기본 점수와 콘텐츠 기반 점수, 로그 기반 보너스 점수를 결합한 하이브리드 점수
        - Top-K Off-Policy Correction 적용
        """
        # 1. 기본 점수 계산 (위치 기반)
        base_scores = self.calculate_base_score(hospitals_df)
        hospitals_df['base_score'] = base_scores
        
        # 2. 콘텐츠 기반 점수 계산
        similarity_scores = cosine_similarity(user_embedding.reshape(1, -1), hospital_embeddings)[0]
        hospitals_df['content_score'] = (similarity_scores + 1) / 2  # -1~1 -> 0~1
        
        # 3. 정책 네트워크를 통한 행동 확률 계산 (REINFORCE)
        with torch.no_grad():
            # 각 병원의 임베딩에 대해 정책 확률 계산
            policy_probs = []
            for hospital_embedding in hospital_embeddings:
                # 사용자 임베딩과 병원 임베딩의 차이를 입력으로 사용
                state = torch.FloatTensor(user_embedding - hospital_embedding)
                prob = self.policy_network(state)
                policy_probs.append(prob.item())
            
            hospitals_df['policy_prob'] = policy_probs
        
        # 4. Top-K Off-Policy Correction 적용
        hospitals_df['importance_weight'] = 1.0 / (hospitals_df['policy_prob'] + 1e-8)
        
        # 5. 보상 계산 및 정렬
        rewards = []
        for idx, row in hospitals_df.iterrows():
            reward = self.calculate_reward(row, user_embedding, hospital_embeddings[idx], member_id)
            rewards.append(reward)
        hospitals_df['reward'] = rewards
        
        # 6. 로그 체크 및 보너스 점수 추가
        if member_id and self.feedback_manager:
            hospital_names = hospitals_df['name'].tolist()
            bonus_scores = self.feedback_manager.check_hospital_logs(member_id, hospital_names)
            
            # 보너스 점수 적용
            for idx, row in hospitals_df.iterrows():
                if row['name'] in bonus_scores:
                    hospitals_df.at[idx, 'reward'] += bonus_scores[row['name']]
        
        # 7. 최종 점수 계산 (보상 * 중요도 가중치) - similarity 컬럼에 저장
        hospitals_df['similarity'] = hospitals_df['reward'] * hospitals_df['importance_weight']
        
        # 8. 상위 K개 선택
        recommended = hospitals_df.nlargest(self.K, 'similarity')
        
        # 9. 불필요한 컬럼 제거
        recommended = recommended.drop(columns=['policy_prob', 'importance_weight', 'reward', 'base_score', 'content_score'])
        
        return recommended

    def update_from_feedback(self, member_id):
        """
        특정 사용자의 피드백 데이터를 기반으로 정책 네트워크 업데이트
        """
        if not self.feedback_manager or not member_id:
            return
            
        # 사용자의 피드백 데이터 가져오기 (Redis 캐시 사용)
        feedback_data = self.feedback_manager.get_hospital_feedback(member_id)
        if feedback_data.empty:
            return
            
        total_loss = 0
        batch_size = 0
        
        # 각 선택된 병원에 대해 정책 업데이트
        for _, row in feedback_data.iterrows():
            # 보상 계산 (Redis 캐시 사용)
            reward = self.feedback_manager.calculate_hospital_reward(row, feedback_data)
            
            if reward > 0:
                # 상태 생성 (사용자 임베딩과 병원 임베딩의 차이)
                state = torch.FloatTensor(row['user_embedding'] - row['facility_embedding'])
                action_prob = self.policy_network(state)
                
                # 손실 함수 계산
                loss = -torch.log(action_prob) * reward
                total_loss += loss
                batch_size += 1
        
        if batch_size > 0:
            # 평균 손실로 업데이트
            avg_loss = total_loss / batch_size
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()
            
        return avg_loss.item() if batch_size > 0 else 0