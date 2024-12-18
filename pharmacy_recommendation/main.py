import pandas as pd
from utils.direction_for_pharmacy import get_travel_time
from utils.es_functions_for_pharmacy import query_pharmacy_elasticsearch

if __name__ == "__main__":
    # 사용자 입력
    user_lat = 37.545216  # 위도 (예: 서울)
    user_lon = 126.964794  # 경도 (예: 서울)


    # Elasticsearch 쿼리 실행
    es_results = query_pharmacy_elasticsearch(user_lat, user_lon)

    # 검색 결과 확인
    if "hits" in es_results and es_results['hits']['total']['value'] > 0:
        print(f"총 {es_results['hits']['total']['value']}개의 약국을 찾았습니다.")

        # 약국 데이터를 DataFrame으로 변환
        pharmacy_data = [hit['_source'] for hit in es_results['hits']['hits']]
        df = pd.DataFrame(pharmacy_data)

        travel_times = []
        for _, row in df.iterrows():
            pharmacy_lat = row['wgs84lat']
            pharmacy_lon = row['wgs84lon']
            travel_time_sec = get_travel_time(user_lat, user_lon, pharmacy_lat, pharmacy_lon)
            travel_times.append(travel_time_sec)

        # 소요시간 추가
        df['travel_time_sec'] = travel_times
        df['travel_time_h'] = df['travel_time_sec'] // 3600
        df['travel_time_min'] = (df['travel_time_sec'] % 3600) // 60
        
        # 결과 출력
        print(df)
    else:
        print("조건에 맞는 약국이 없습니다.")