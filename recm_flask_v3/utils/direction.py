# Google Directions API 사용 -> 모드별 소요 시간 및 거리 측정
def get_travel_time_and_distance(user_lat, user_lon, hospital_lat, hospital_lon):
    import requests
    import os
    """
    Google Directions API를 사용하여 사용자 위치와 병원 간의 이동 소요시간과 거리를 반환합니다.

    Parameters:
    - user_lat (float): 사용자 위도
    - user_lon (float): 사용자 경도
    - hospital_lat (float): 병원 위도
    - hospital_lon (float): 병원 경도

    Returns:
    - dict: 모드별 소요 시간 (초 단위) 및 거리 (km)
    """
    
    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": f"{user_lat},{user_lon}",
        "destination": f"{hospital_lat},{hospital_lon}",
        "key": os.getenv("GOOGLE_API_KEY"),
        "mode": "transit"
    }

    response = requests.get(url, params=params)
    results = {}

    if response.status_code == 200:
        data = response.json()
        if "routes" in data and data["routes"]:
            leg = data["routes"][0]["legs"][0]
            results["transit_travel_time_sec"] = leg["duration"]["value"]
            results["transit_travel_distance_km"] = leg["distance"]["value"] / 1000
        else:
            results["transit_travel_time_sec"] = None
            results["transit_travel_distance_km"] = None
    else:
        results["transit_travel_time_sec"] = None
        results["transit_travel_distance_km"] = None

    return results


# 행 데이터에 대한 여행 시간 및 거리 계산
def calculate_travel_time_and_distance(row, user_lat, user_lon):
    import time
    from .geocode import address_to_coords
    try:
        hospital_lat = row["latitude"]
        hospital_lon = row["longitude"]
        addr = row["address"]

        # 병원 좌표가 없을 경우 주소를 통해 좌표를 찾음
        if hospital_lat is None or hospital_lon is None:
            coords = address_to_coords(addr)
            if "lat" in coords and "lon" in coords:
                hospital_lat, hospital_lon = coords["lat"], coords["lon"]

        travel_data = get_travel_time_and_distance(user_lat, user_lon, hospital_lat, hospital_lon)

        # 시간 데이터를 시/분/초로 변환
        if travel_data.get("transit_travel_time_sec") is not None:
            travel_time_sec = travel_data["transit_travel_time_sec"]
            travel_data["transit_travel_time_h"] = travel_time_sec // 3600
            travel_data["transit_travel_time_m"] = (travel_time_sec % 3600) // 60
            travel_data["transit_travel_time_s"] = travel_time_sec % 60
        else:
            travel_data["transit_travel_time_h"] = None
            travel_data["transit_travel_time_m"] = None
            travel_data["transit_travel_time_s"] = None

        time.sleep(0.3)  # API 호출 제한 준수
        return travel_data  # dict with travel times and distances
    except Exception as e:
        print(f"Error calculating travel time for hospital {row['name']}: {e}")
        return None