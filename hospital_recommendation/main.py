import pandas as pd
import numpy as np
import configparser
from utils.es_functions import query_elasticsearch, filtering  #Elasticsearch 관련 모듈
from utils.recommendation import HospitalRecommender
from utils.direction import get_travel_time
from utils.geocode import address_to_coords
import time

if __name__ == "__main__":
    # ConfigParser 초기화
    config = configparser.ConfigParser()
    # keys.config 파일 읽기
    config.read('C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config')

    # 사용자 정보 입력(sample)(나중에 spring에서 받아오게끔 바꿔야 함)
    basic_info = {
        "language": "Mongolian",
        "number": "010-1234-5678",
        "address": "서울특별시 강남구 테헤란로 427",
        "gender": "여성",
        "age": 35,
        "height": 175,
        "weight": 70
    }
    health_info = {
        "pastHistory": "고혈압",
        "familyHistory": "심장병",
        "nowMedicine": "아스피린",
        "allergy": "페니실린"
    }

    # HospitalRecommender 인스턴스 생성
    recommender = HospitalRecommender()
    
    # 카카오 REST API 키
    api_key = config['API_KEYS']['kakao_api_key']
    coords = address_to_coords(basic_info['address'], api_key)

    user_lat = coords['lat']  
    user_lon = coords['lon']
    department = "내과" #gpt 통해서 prompt 받고 결과 확인해야 함.
    suspected_disease = "감기"

    #선택 옵션을 백에서 확인해야 함.
    secondary_hospital = False
    tertiary_hospital = False


    # Elasticsearch 쿼리 실행
    es_results = query_elasticsearch(user_lat, user_lon, department, secondary_hospital, tertiary_hospital)

    # 검색 결과 확인
    if "hits" in es_results and len(es_results['hits']['hits']) > 0:
        print(f"총 {len(es_results['hits']['hits'])}개의 병원을 찾았습니다.")
        
        # 필터링된 병원 데이터를 추출
        filtered_hospitals = filtering(es_results)
        hospital_data = [hospital for hospital in filtered_hospitals]

        # DataFrame으로 변환
        df = pd.DataFrame(hospital_data)

        # 제외할 열 설정
        excluded_columns = ['@version']
        df = df.drop(columns=[col for col in excluded_columns if col in df.columns], errors='ignore')

        # 열 정렬
        column_order = [
            'id', 'name', 'addr', 'department', 'distance_in_m', 'telephone',
            'latitude', 'longitude', 'sort_score'
        ]
        # 열 정렬 및 누락된 열은 자동으로 뒤에 추가
        df = df[[col for col in column_order if col in df.columns] + [col for col in df.columns if col not in column_order]]

        error_occurred = False  # 오류 발생 여부 확인용 변수
        processed_count = 0  # 성공적으로 처리된 데이터 개수
        

        sampled_df = df[:100]
        # 각 병원까지의 소요시간을 계산하여 데이터프레임에 추가
        travel_times = []
        for _, row in sampled_df.iterrows():
            try:
                hospital_lat = row['latitude']
                hospital_lon = row['longitude']
                retry_count = 0

                while retry_count < 3:  # 최대 3번 재시도
                    try:
                        travel_time_sec = get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon)
                        if travel_time_sec is not None:
                            travel_times.append(travel_time_sec)
                            processed_count += 1
                            time.sleep(0.3)  # 초당 요청 수 제한 준수 (TPS가 5라면 최소 0.3초 간격)
                            break  # 성공하면 반복 종료
                        else:
                            raise ValueError("소요 시간을 가져오는 데 실패했습니다.")
                    except Exception as e:
                        if "429" in str(e):  # HTTP 429 에러 처리
                            retry_count += 1
                            wait_time = 60 if retry_count == 3 else 2 ** retry_count  # 마지막 재시도에서 60초 대기
                            print(f"HTTP 요청 오류: 429. {wait_time}초 대기 후 재시도합니다... (재시도 {retry_count}/3)")
                            time.sleep(wait_time)
                        else:
                            raise e  # 다른 에러는 다시 raise
            except Exception as e:
                print(f"에러 발생: {e}, 지금까지 처리된 데이터: {processed_count}개")
                error_occurred = True
                break  # 오류 발생 시 반복문 종료

        # travel_times에 따라 항상 데이터프레임 업데이트
        sampled_df = sampled_df.iloc[:len(travel_times)]  # 이미 처리된 데이터만 반영
        sampled_df['travel_time_sec'] = travel_times
        sampled_df['travel_time_h'] = sampled_df['travel_time_sec'] // 3600
        sampled_df['travel_time_min'] = (sampled_df['travel_time_sec'] % 3600) // 60

        # 결과 출력
        if error_occurred:
            print(f"오류로 인해 병원 데이터 처리가 중단되었습니다. 성공적으로 처리된 데이터: {processed_count}개")
        else:
            print("조건에 맞는 병원이 처리되었습니다.")

    hospitals_df = sampled_df.copy()
    # 임베딩 생성
    user_embedding = recommender.embed_user_profile(basic_info, health_info)
    hospital_embeddings = recommender.embed_hospital_data(hospitals_df)

    #autoencoder = recommender.train_autoencoder(hospital_embeddings, input_dim=hospital_embeddings.shape[1], latent_dim=64)
    vae = recommender.train_vae(hospital_embeddings, input_dim=hospital_embeddings.shape[1], latent_dim=64, hidden_dim=128)

    recommended_hospitals = recommender.recommend_hospitals(
        vae = vae,
        #autoencoder=autoencoder, 
        user_embedding=user_embedding, 
        hospital_embeddings=hospital_embeddings, 
        hospitals_df=hospitals_df, 
        department=department,
        #use_autoencoder=False
        use_vae = True
    )

    print(recommended_hospitals[["name", "clcdnm", "department", "travel_time_h", "travel_time_min", "similarity"]])