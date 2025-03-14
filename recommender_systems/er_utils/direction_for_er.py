def get_travel_time_er(user_lat, user_lon, hospital_lat, hospital_lon):
    import requests
    import configparser

    config = configparser.ConfigParser()
    config.read('keys.config')  #Google API 키 파일

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{user_lat},{user_lon}",
        "destination": f"{hospital_lat},{hospital_lon}",
        "key": config['API_KEYS']['google_api_key'],
        "mode": "transit"
    }
    results = {}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        print(data)
        if "routes" in data and data["routes"]:
            leg = data["routes"][0]["legs"][0]
            travel_time_sec = leg["duration"]["value"]
            distance_km = leg["distance"]["value"] / 1000
            results["transit_travel_time_sec"] = travel_time_sec
            results["transit_travel_distance_km"] = distance_km
        else:
            results["transit_travel_time_sec"] = None
            results["transit_travel_distance_km"] = None
    else:
        results["transit_travel_time_sec"] = None
        results["transit_travel_distance_km"] = None
    
    return results

def calculate_travel_time_and_sort(enriched_df, user_lat, user_lon):
    import pandas as pd

    enriched_df = enriched_df.copy()
    #print(enriched_df)

    def enrich_row_with_transit(row):
        if row["wgs84Lat"] and row["wgs84Lon"]:
            #있음!
            infos = get_travel_time_er(
                user_lat, user_lon,
                float(row["wgs84Lat"]),
                float(row["wgs84Lon"])
            )

            return pd.Series({
                "transit_travel_time_sec": infos.get("transit_travel_time_sec"),
                "transit_travel_distance_km": infos.get("transit_travel_distance_km"),
            })
        else:
            return pd.Series({
                "transit_travel_time_sec": None,
                "transit_travel_distance_km": None,
            })
    #Apply enrichment-이게 None이어서 문제 발생!!
    enriched_df[["transit_travel_time_sec", "transit_travel_distance_km"]] = enriched_df.apply(enrich_row_with_transit, axis=1)
    
    #시간 변환 (시, 분, 초)
    enriched_df["transit_travel_time_h"] = enriched_df["transit_travel_time_sec"].fillna(0).astype(int) // 3600
    enriched_df["transit_travel_time_m"] = (enriched_df["transit_travel_time_sec"].fillna(0).astype(int) % 3600) // 60
    enriched_df["transit_travel_time_s"] = enriched_df["transit_travel_time_sec"].fillna(0).astype(int) % 60

    #transit_travel_time_sec와 hvec 기준으로 정렬
    enriched_df["hvec_abs"] = enriched_df["hvec"].astype(float).abs()
    
    enriched_df.sort_values(
        by=["transit_travel_time_sec", "hvec_abs"],
        ascending=[True, True],  #대중교통 소요시간: 오름차순, hvec 절대값: 오름차순
        inplace=True
    )

    #hvec_abs 컬럼 삭제 (정렬에만 사용)
    enriched_df.drop(columns=["hvec_abs"], inplace=True)

    return enriched_df
