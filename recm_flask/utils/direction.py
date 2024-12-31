# 네이버 길찾기 api 사용 -> 소요 시간 측정
def get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon):
    import requests
    import os
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
        "X-NCP-APIGW-API-KEY-ID": os.getenv("NAVER_API_KEY_ID"),
        "X-NCP-APIGW-API-KEY": os.getenv("NAVER_API_KEY")
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