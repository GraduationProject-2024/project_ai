import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import redis
import json
from utils.for_redis import get_redis_client

class FeedbackManager:
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        self.redis_client = get_redis_client()#redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 604800  # 7일

    def get_hospital_feedback(self, member_id):
        """병원 선택 피드백 데이터 수집 (Redis 캐싱)"""
        cache_key = f"hospital_feedback:{member_id}"
        
        # Redis에서 캐시된 데이터 확인
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return pd.DataFrame(json.loads(cached_data))
            
        if not self.db_connection:
            return pd.DataFrame()
            
        query = """
        SELECT 
            sh.hpid,
            sh.selected_at,
            h.name as hospital_name,
            h.department,
            h.clcdnm as hospital_type
        FROM selected_hp sh
        JOIN hospital h ON sh.hpid = h.hpid
        WHERE sh.member_id = %s
        ORDER BY sh.selected_at DESC
        """
        
        try:
            df = pd.read_sql(query, self.db_connection, params=(member_id,))
            # Redis에 캐싱
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(df.to_dict(orient='records'))
            )
            return df
        except Exception as e:
            print(f"Error getting hospital feedback: {e}")
            return pd.DataFrame()

    def get_pharmacy_feedback(self, member_id):
        """약국 선택 피드백 데이터 수집 (Redis 캐싱)"""
        cache_key = f"pharmacy_feedback:{member_id}"
        
        # Redis에서 캐시된 데이터 확인
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return pd.DataFrame(json.loads(cached_data))
            
        if not self.db_connection:
            return pd.DataFrame()
            
        query = """
        SELECT 
            sp.hpid,
            sp.selected_at,
            p.dutyname as pharmacy_name,
            p.dutyaddr as address
        FROM selected_ph sp
        JOIN pharmacy p ON sp.hpid = p.hpid
        WHERE sp.member_id = %s
        ORDER BY sp.selected_at DESC
        """
        
        try:
            df = pd.read_sql(query, self.db_connection, params=(member_id,))
            # Redis에 캐싱
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(df.to_dict(orient='records'))
            )
            return df
        except Exception as e:
            print(f"Error getting pharmacy feedback: {e}")
            return pd.DataFrame()

    def calculate_hospital_reward(self, hospital_data, feedback_data):
        """병원 선택에 대한 보상 계산"""
        if feedback_data.empty:
            return 0.0
            
        # 최근 선택된 병원인 경우 높은 보상
        recent_selections = feedback_data[
            feedback_data['selected_at'] > datetime.now() - timedelta(days=7)
        ]
        
        if hospital_data['name'] in recent_selections['hospital_name'].values:
            return 0.2  # 최근 선택된 병원에 대한 보상
            
        return 0.0

    def calculate_pharmacy_reward(self, pharmacy_data, feedback_data):
        """약국 선택에 대한 보상 계산"""
        if feedback_data.empty:
            return 0.0
            
        # 최근 선택된 약국인 경우 높은 보상
        recent_selections = feedback_data[
            feedback_data['selected_at'] > datetime.now() - timedelta(days=7)
        ]
        
        if pharmacy_data['dutyname'] in recent_selections['pharmacy_name'].values:
            return 0.2  # 최근 선택된 약국에 대한 보상
            
        return 0.0

    def check_hospital_logs(self, member_id, hospital_names):
        """병원 로그 확인 및 보너스 점수 반환 (Redis 캐싱)"""
        cache_key = f"hospital_bonus:{member_id}"
        
        # Redis에서 캐시된 보너스 점수 확인
        cached_bonus = self.redis_client.get(cache_key)
        if cached_bonus:
            bonus_scores = json.loads(cached_bonus)
            return {name: score for name, score in bonus_scores.items() if name in hospital_names}
            
        if not self.db_connection:
            return {}
            
        try:
            # 회원의 병원 로그 조회
            query = """
            SELECT h.name as hospital_name, sh.id as selected_id
            FROM hospital h
            LEFT JOIN selected_hp sh ON h.hpid = sh.hpid AND sh.member_id = %s
            WHERE h.name IN %s
            """
            df = pd.read_sql(query, self.db_connection, params=(member_id, tuple(hospital_names)))
            
            # 보너스 점수 계산
            bonus_scores = {}
            for _, row in df.iterrows():
                if row['selected_id'] is not None:
                    bonus_scores[row['hospital_name']] = 0.2
                    
            # Redis에 캐싱
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(bonus_scores)
            )
            
            return bonus_scores
        except Exception as e:
            print(f"병원 로그 확인 중 오류 발생: {e}")
            return {}

    def check_pharmacy_logs(self, member_id, pharmacy_names):
        """약국 로그 확인 및 보너스 점수 반환 (Redis 캐싱)"""
        cache_key = f"pharmacy_bonus:{member_id}"
        
        # Redis에서 캐시된 보너스 점수 확인
        cached_bonus = self.redis_client.get(cache_key)
        if cached_bonus:
            bonus_scores = json.loads(cached_bonus)
            return {name: score for name, score in bonus_scores.items() if name in pharmacy_names}
            
        if not self.db_connection:
            return {}
            
        try:
            # 회원의 약국 로그 조회
            query = """
            SELECT p.dutyname as pharmacy_name, sp.id as selected_id
            FROM pharmacy p
            LEFT JOIN selected_ph sp ON p.hpid = sp.hpid AND sp.member_id = %s
            WHERE p.dutyname IN %s
            """
            df = pd.read_sql(query, self.db_connection, params=(member_id, tuple(pharmacy_names)))
            
            # 보너스 점수 계산
            bonus_scores = {}
            for _, row in df.iterrows():
                if row['selected_id'] is not None:
                    bonus_scores[row['pharmacy_name']] = 0.2
                    
            # Redis에 캐싱
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(bonus_scores)
            )
            
            return bonus_scores
        except Exception as e:
            print(f"약국 로그 확인 중 오류 발생: {e}")
            return {}

    def check_operating_hours(self, pharmacy_data):
        """
        현재 시간에 약국이 영업 중인지 확인하고 보너스 점수 반환
        """
        try:
            current_time = datetime.now()
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_weekday = current_time.weekday() + 1  # 1-7 (월-일)

            # 현재 시간을 자정 이후 분으로 변환
            current_time_minutes = current_hour * 60 + current_minute

            # 현재 요일의 영업시간 확인
            start_time_key = f"{current_weekday}s"
            end_time_key = f"{current_weekday}c"

            if start_time_key in pharmacy_data and end_time_key in pharmacy_data:
                start_time = pharmacy_data[start_time_key]
                end_time = pharmacy_data[end_time_key]

                if start_time and end_time:
                    # 약국 영업시간을 자정 이후 분으로 변환
                    start_hour = int(start_time[:2])
                    start_minute = int(start_time[2:])
                    end_hour = int(end_time[:2])
                    end_minute = int(end_time[2:])

                    start_minutes = start_hour * 60 + start_minute
                    end_minutes = end_hour * 60 + end_minute

                    # 현재 시간이 영업시간 내인지 확인
                    if start_minutes <= current_time_minutes <= end_minutes:
                        return 0.3  # 영업 중인 경우 보너스

            return 0.0
        except Exception as e:
            print(f"영업시간 확인 중 오류 발생: {e}")
            return 0.0 