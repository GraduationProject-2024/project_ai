# 네이버 길찾기 api 사용 -> 소요 시간 측정
def get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon):
    import requests
    import configparser

    # ConfigParser 초기화
    config = configparser.ConfigParser()

    # keys.config 파일 읽기
    config.read('C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config')

    """
    네이버 지도 Directions API를 사용하여 사용자 위치와 병원 간의 자동차 이동 소요시간을 반환합니다.

    Parameters:
    - user_lat (float): 사용자 위도
    - user_lon (float): 사용자 경도
    - hospital_lat (float): 병원 위도
    - hospital_lon (float): 병원 경도

    Returns:
    - int: 이동 소요시간 (초 단위)
    """

    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": config['API_KEYS']['naver_api_key_id'],
        "X-NCP-APIGW-API-KEY": config['API_KEYS']['naver_api_key']
    }
    params = {
        "start": f"{user_lon},{user_lat}",
        "goal": f"{hospital_lon},{hospital_lat}",
        "option": "trafast"  # 가장 빠른 경로
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['code'] == 0:
            # 총 소요시간을 초 단위로 반환
            return data['route']['trafast'][0]['summary']['duration'] / 1000
        else:
            print(f"API 응답 오류: {data['message']}")
            return None
    else:
        print(f"HTTP 요청 오류: {response.status_code}")
        return None

# enriched_df에 소요시간 계산 및 정렬
def calculate_travel_time_and_sort(enriched_df, user_lat, user_lon):
    enriched_df = enriched_df.copy()

    # 소요시간 컬럼 추가
    enriched_df["travelTime"] = enriched_df.apply(
        lambda row: get_travel_time(
            user_lat, user_lon,
            float(row["wgs84Lat"]) if row["wgs84Lat"] else None,
            float(row["wgs84Lon"]) if row["wgs84Lon"] else None
        ) if row["wgs84Lat"] and row["wgs84Lon"] else None,
        axis=1
    )

    # 소요시간과 hvec 기준으로 정렬
    enriched_df["hvec_abs"] = enriched_df["hvec"].astype(float).abs()
    enriched_df.sort_values(
        by=["travelTime", "hvec_abs"],
        ascending=[True, True],  # 소요시간: 오름차순, hvec 절대값: 오름차순
        inplace=True
    )

    # hvec_abs 컬럼 삭제 (정렬에만 사용)
    enriched_df.drop(columns=["hvec_abs"], inplace=True)

    return enriched_df