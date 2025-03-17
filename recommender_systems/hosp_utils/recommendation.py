import openai
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

class HospitalRecommender:
    def __init__(self, model_name='text-embedding-3-small'):
        self.model_name = model_name  #OpenAI Embedding Model 사용
        self.scaler = MinMaxScaler(feature_range=(0, 1))

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
        unique_sentences = list(set(hospital_sentences))  # 중복 제거
        unique_embeddings = self.get_embedding(unique_sentences, batch_size=256)
        print(unique_embeddings.shape)
        #4️.병원명 기준으로 임베딩 매핑
        embedding_dict = dict(zip(unique_sentences, unique_embeddings))
        hospital_embeddings = np.array([embedding_dict[sentence] for sentence in hospital_sentences])

        return hospital_embeddings

    
    from sklearn.metrics.pairwise import cosine_similarity
    def recommend_hospitals(self, user_embedding, hospital_embeddings, hospitals_df):
        # 유사도 계산을 NumPy 벡터 연산으로 최적화
        similarities = np.dot(hospital_embeddings, user_embedding) / (
            np.linalg.norm(hospital_embeddings, axis=1) * np.linalg.norm(user_embedding) + 1e-8
        )
        #(0~1 범위)
        
        similarities = self.scaler.fit_transform(similarities.reshape(-1, 1)).flatten()
        
        hospitals_df["similarity"] = similarities
        return hospitals_df