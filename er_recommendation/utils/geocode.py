# 사용자 주소 -> 좌표값으로 변환하는 geocode 함수
def address_to_coords(address, api_key):
    import requests
    """
    주소를 사용하여 카카오 로컬 API를 통해 위도와 경도로 변환합니다.

    Parameters:
    - address (str): 도로명 주소
    - api_key (str): 카카오 REST API 키

    Returns:
    - dict: 변환된 좌표 정보 {"lat": 위도, "lon": 경도}
    """
    url = "https://dapi.kakao.com/v2/local/search/address.json"
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