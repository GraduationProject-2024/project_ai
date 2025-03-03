# 사용자 주소 -> 좌표값으로 변환하는 geocode 함수
def address_to_coords(address):
    import requests
    import configparser

    config = configparser.ConfigParser()
    config.read('keys.config')

    """
    주소를 사용하여 카카오 로컬 API를 통해 위도와 경도로 변환합니다.

    Parameters:
    - address (str): 도로명 주소
    - api_key (str): 카카오 REST API 키

    Returns:
    - dict: 변환된 좌표 정보 {"lat": 위도, "lon": 경도}
    """
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    api_key = config['API_KEYS']['kakao_api_key']
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            coords = result['documents'][0]['address']
            return {"lat": float(coords['y']), "lon": float(coords['x'])}
        else:
            return {"error": "주소 정보를 찾을 수 없습니다."}
    else:
        return {"error": f"에러 발생: {response.status_code}"}

# 좌표값 -> 주소로 변환하는 geocode 함수
def coords_to_address(lat, lon):
    import requests
    import configparser

    config = configparser.ConfigParser()
    config.read('keys.config')

    """
    위도와 경도를 사용하여 카카오 로컬 API를 통해 주소로 변환합니다.

    Parameters:
    - lat (float): 위도
    - lon (float): 경도
    - api_key (str): 카카오 REST API 키

    Returns:
    - dict: 변환된 주소 정보 {"address_name": 주소}
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    api_key = config['API_KEYS']['kakao_api_key']
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"x": lon, "y": lat}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            address = result['documents'][0]['address']
            return {"address_name": address['address_name']}
        else:
            return {"error": "좌표에 해당하는 주소 정보를 찾을 수 없습니다."}
    else:
        return {"error": f"에러 발생: {response.status_code}"}
