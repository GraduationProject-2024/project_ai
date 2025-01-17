def get_travel_time_er(user_lat, user_lon, hospital_lat, hospital_lon):
    import requests
    import os

    # 혼잡도 매핑
    congestion_mapping = {
        1: "원활",
        2: "서행",
        3: "지체",
        4: "정체"
    }

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
            # 'trafast' 키 확인
            if 'trafast' in data['route'] and data['route']['trafast']:
                summary = data['route']['trafast'][0]['summary']
                travel_time_sec = summary['duration'] / 1000  # 초 단위로 변환
                distance_km = summary['distance'] / 1000      # m -> km 단위 변환

                # 혼잡도 값 처리
                sections = data['route']['trafast'][0].get('section', [])
                congestion_values = [section.get('congestion', None) for section in sections if 'congestion' in section]

                if congestion_values:
                    # 혼잡도 값 평균 계산 후 가장 가까운 정수로 변환
                    avg_congestion = round(sum(congestion_values) / len(congestion_values))
                    congestion_text = congestion_mapping.get(avg_congestion, "알 수 없음")  # 텍스트 매핑
                else:
                    congestion_text = "알 수 없음"

                return {
                    "travel_time_sec": travel_time_sec,
                    "distance_km": distance_km,
                    "congestion": congestion_text
                }
            else:
                print("유효한 trafast 경로 데이터가 없습니다.")
                return {"travel_time_sec": None, "distance_km": None, "congestion": "알 수 없음"}
        else:
            print(f"API 응답 오류: {data['message']}")
            return {"travel_time_sec": None, "distance_km": None, "congestion": "알 수 없음"}
    else:
        print(f"HTTP 요청 오류: {response.status_code}")
        return {"travel_time_sec": None, "distance_km": None, "congestion": "알 수 없음"}


def calculate_travel_time_and_sort(enriched_df, user_lat, user_lon):
    import pandas as pd
    enriched_df = enriched_df.copy()

    # 소요시간, 거리 및 혼잡도 컬럼 추가
    enriched_df[["travelTime", "distance_km", "congestion"]] = enriched_df.apply(
        lambda row: pd.Series(get_travel_time_er(
            user_lat, user_lon,
            float(row["wgs84Lat"]) if row["wgs84Lat"] else None,
            float(row["wgs84Lon"]) if row["wgs84Lon"] else None
        )) if row["wgs84Lat"] and row["wgs84Lon"] else {"travel_time_sec": None, "distance_km": None, "congestion": None},
        axis=1
    )

    # travel_time_sec 확인 후 travelTime에 값 복사
    if "travelTime" not in enriched_df.columns:
        enriched_df["travelTime"] = enriched_df["travel_time_sec"]

    # 'travelTime' 값을 '시', '분', '초'로 변환하여 추가
    enriched_df["travel_time_h"] = enriched_df["travelTime"].fillna(0).astype(int) // 3600
    enriched_df["travel_time_m"] = (enriched_df["travelTime"].fillna(0).astype(int) % 3600) // 60
    enriched_df["travel_time_s"] = enriched_df["travelTime"].fillna(0).astype(int) % 60

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
