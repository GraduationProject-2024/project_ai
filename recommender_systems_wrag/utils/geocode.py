#사용자 주소 -> 좌표값으로 변환하는 geocode 함수(카카오오)
def address_to_coords(address):
    import requests
    import configparser

    config=configparser.ConfigParser()
    config.read('keys.config')

    url="https://dapi.kakao.com/v2/local/search/address.json"
    api_key=config['API_KEYS']['kakao_api_key']
    headers={"Authorization": f"KakaoAK {api_key}"}
    params={"query": address}

    response=requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result=response.json()
        if result['documents']:
            coords=result['documents'][0]['address']
            return {"lat": float(coords['y']), "lon": float(coords['x'])}
        else:
            return {"error": "주소 정보를 찾을 수 없습니다."}
    else:
        return {"error": f"에러 발생: {response.status_code}"}

#좌표값 -> 주소
def coords_to_address(lat, lon):
    import requests
    import configparser

    config=configparser.ConfigParser()
    config.read('keys.config')

    url="https://dapi.kakao.com/v2/local/geo/coord2address.json"
    api_key=config['API_KEYS']['kakao_api_key']
    headers={"Authorization": f"KakaoAK {api_key}"}
    params={"x": lon, "y": lat}

    response=requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        result=response.json()
        if result['documents']:
            address=result['documents'][0]['address']
            return {"address_name": address['address_name']}
        else:
            return {"error": "좌표에 해당하는 주소 정보를 찾을 수 없습니다."}
    else:
        return {"error": f"에러 발생: {response.status_code}"}
