# 네이버 길찾기 api 사용 -> 소요 시간 측정
def get_travel_time(user_lat, user_lon, pharmacy_lat, pharmacy_lon):
    import requests
    import configparser

    # ConfigParser 초기화
    config = configparser.ConfigParser()

    # keys.config 파일 읽기
    config.read('../keys.config')

    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": config['API_KEYS']['naver_api_key_id'],
        "X-NCP-APIGW-API-KEY": config['API_KEYS']['naver_api_key']
    }
    params = {
        "start": f"{user_lon},{user_lat}",
        "goal": f"{pharmacy_lon},{pharmacy_lat}",
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