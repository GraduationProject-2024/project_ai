# 네이버 길찾기 api 사용 -> 소요 시간 측정
def get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon, addr):
    import requests
    from .geocode import address_to_coords
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
            #print(f"API 응답 오류: {data['message']}")
            if data['message'] == "출발지와 도착지가 동일합니다. 확인 후 다시 지정해주세요.":
                print(f"API 응답 오류: {data['message']}")
                return 0
            if data['message'] == "출발지 또는 도착지가 도로 주변이 아닙니다. 위치를 변경해 주세요.":
                if hospital_lat is None or hospital_lon is None:
                    # 좌표가 None이면 addr을 사용하여 geocoding
                    coords = address_to_coords(addr)
                    if "lat" in coords and "lon" in coords:
                        hospital_lat, hospital_lon = coords["lat"], coords["lon"]
                        # 요청을 다시 시도
                        params["goal"] = f"{hospital_lon},{hospital_lat}"
                        retry_response = requests.get(url, headers=headers, params=params)
                        if retry_response.status_code == 200:
                            retry_data = retry_response.json()
                            if retry_data['code'] == 0:
                                return retry_data['route']['trafast'][0]['summary']['duration'] / 1000
                # 그래도 실패하면 None 반환
                print(f"API 응답 오류: {data['message']}")
                return None
            print(f"API 응답 오류: {data['message']}")
            return None
    else:
        print(f"HTTP 요청 오류: {response.status_code}")
        return None

def calculate_travel_time(row, user_lat, user_lon):
        import time
        try:
            hospital_lat = row["latitude"]
            hospital_lon = row["longitude"]
            addr = row["address"]
            travel_time_sec = get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon, addr)
            time.sleep(0.3)  # API 호출 제한 준수
            return travel_time_sec or 0
        except Exception as e:
            print(f"Error calculating travel time for hospital {row['name']}: {e}")
            return None