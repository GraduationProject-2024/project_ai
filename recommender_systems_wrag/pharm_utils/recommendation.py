import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime, timedelta
from utils.feedback_manager import FeedbackManager

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):  # input_dim: [이동시간, 기본점수]
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class PharmacyRecommender:
    def __init__(self, db_connection=None):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.policy_network = PolicyNetwork()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        self.gamma = 0.99
        self.K = 15
        self.feedback_manager = FeedbackManager(db_connection) if db_connection else None

    def update_from_feedback(self, member_id):
        """
        특정 사용자의 피드백 데이터를 기반으로 정책 네트워크 업데이트
        """
        if not self.feedback_manager or not member_id:
            return
            
        # 사용자의 피드백 데이터 가져오기 (Redis 캐시 사용)
        feedback_data = self.feedback_manager.get_pharmacy_feedback(member_id)
        if feedback_data.empty:
            return
            
        total_loss = 0
        batch_size = 0
        
        # 각 선택된 약국에 대해 정책 업데이트
        for _, row in feedback_data.iterrows():
            # 보상 계산 (Redis 캐시 사용)
            reward = self.feedback_manager.calculate_pharmacy_reward(row, feedback_data)
            
            if reward > 0:
                # 약국의 특성으로 상태 구성 [이동시간, 기본점수]
                features = torch.FloatTensor([
                    1 / (1 + self.calculate_base_score(pd.DataFrame([row]))[0]),  # 이동시간 (정규화)
                    self.calculate_base_score(pd.DataFrame([row]))[0]  # 기본점수
                ])
                
                action_prob = self.policy_network(features)
                
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

    def calculate_base_score(self, pharmacy_df):
        """
        기본 점수 계산 (위치 기반 필터링)
        - 이동 시간을 기반으로 한 거리 기반 점수
        """
        # 이동 시간을 초 단위로 변환
        total_travel_time = (
            pharmacy_df['transit_travel_time_h'] * 3600 +
            pharmacy_df['transit_travel_time_m'] * 60 +
            pharmacy_df['transit_travel_time_s']
        )
        
        # 이동 시간이 짧을수록 높은 점수
        base_scores = 1 / (1 + total_travel_time)
        return base_scores

    def calculate_reward(self, pharmacy_data, member_id=None):
        """
        약국 데이터를 기반으로 보상 계산
        - 위치 기반 필터링과 영업시간을 결합한 하이브리드 보상 함수
        - 장기적 보상에 gamma 적용
        - 보상 정규화 적용
        """
        # 1. 이동 시간 기반 보상 (위치 기반 필터링) - 70% 가중치
        time_reward = self.calculate_base_score(pd.DataFrame([pharmacy_data])).iloc[0]  # 0~1 범위
        
        # 2. 영업시간 기반 보상 (추가 특성) - 30% 가중치
        operating_hours_bonus = self.feedback_manager.check_operating_hours(pharmacy_data) if self.feedback_manager else 0.0
        # 영업시간 보너스는 이미 0~0.3 범위
        
        # 3. 즉각적인 보상 계산 (0~1 범위)
        immediate_reward = 0.7 * time_reward + operating_hours_bonus
        
        # 4. 장기적 보상 계산 (이전 선택 기록 기반)
        if member_id and self.feedback_manager:
            feedback_data = self.feedback_manager.get_pharmacy_feedback(member_id)
            if not feedback_data.empty:
                long_term_reward = self.feedback_manager.calculate_pharmacy_reward(pharmacy_data, feedback_data)
                # 장기적 보상을 0~1 범위로 정규화
                long_term_reward = min(long_term_reward, 1.0)
                # gamma를 사용하여 장기적 보상 할인
                return immediate_reward + self.gamma * long_term_reward
        
        return immediate_reward

    def recommend_pharmacies(self, pharmacies_df, member_id=None):
        """
        하이브리드 추천 시스템 (REINFORCE + 위치 기반 필터링)
        - REINFORCE 알고리즘을 통한 정책 학습
        - 이동시간과 영업시간을 고려한 위치 기반 필터링
        - Top-K Off-Policy Correction 적용
        """
        # 1. 기본 점수 계산 (위치 기반)
        base_scores = self.calculate_base_score(pharmacies_df)
        pharmacies_df['base_score'] = base_scores
        
        # 2. 정책 네트워크를 통한 행동 확률 계산 (REINFORCE)
        with torch.no_grad():
            # 입력 특성 준비 [이동시간, 기본점수]
            features = torch.FloatTensor(np.column_stack((
                1 / (1 + base_scores),  # 이동시간 (정규화)
                base_scores  # 기본점수
            )))
            action_probs = self.policy_network(features)
        
        # 3. Top-K Off-Policy Correction 적용
        pharmacies_df['policy_prob'] = action_probs.numpy()
        pharmacies_df['importance_weight'] = 1.0 / (pharmacies_df['policy_prob'] + 1e-8)
        
        # 4. 보상 계산
        pharmacies_df['reward'] = pharmacies_df.apply(
            lambda row: self.calculate_reward(row, member_id), axis=1
        )
        
        # 5. 로그 체크 및 보너스 점수 추가
        if member_id and self.feedback_manager:
            pharmacy_names = pharmacies_df['dutyname'].tolist()
            bonus_scores = self.feedback_manager.check_pharmacy_logs(member_id, pharmacy_names)
            
            # 보너스 점수 적용
            for idx, row in pharmacies_df.iterrows():
                if row['dutyname'] in bonus_scores:
                    pharmacies_df.at[idx, 'reward'] += bonus_scores[row['dutyname']]
        
        # 6. 최종 점수 계산 (보상 * 중요도 가중치) - similarity 컬럼에 저장
        pharmacies_df['similarity'] = pharmacies_df['reward'] * pharmacies_df['importance_weight']
        
        # 7. 상위 K개 선택
        recommended = pharmacies_df.nlargest(self.K, 'similarity')
        
        # 8. 불필요한 컬럼 제거
        recommended = recommended.drop(columns=['policy_prob', 'importance_weight', 'reward', 'base_score'])
        
        return recommended

    def update_policy(self, member_id):
        """
        특정 사용자의 피드백 데이터를 기반으로 정책 네트워크 업데이트
        """
        if not self.feedback_manager or not member_id:
            return
            
        # 사용자의 피드백 데이터 가져오기 (Redis 캐시 사용)
        feedback_data = self.feedback_manager.get_pharmacy_feedback(member_id)
        if feedback_data.empty:
            return
            
        total_loss = 0
        batch_size = 0
        
        # 각 선택된 약국에 대해 정책 업데이트
        for _, row in feedback_data.iterrows():
            # 보상 계산 (Redis 캐시 사용)
            reward = self.feedback_manager.calculate_pharmacy_reward(row, feedback_data)
            
            if reward > 0:
                # 약국의 특성으로 상태 구성 [이동시간, 기본점수]
                features = torch.FloatTensor([
                    1 / (1 + self.calculate_base_score(pd.DataFrame([row]))[0]),  # 이동시간 (정규화)
                    self.calculate_base_score(pd.DataFrame([row]))[0]  # 기본점수
                ])
                
                action_prob = self.policy_network(features)
                
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