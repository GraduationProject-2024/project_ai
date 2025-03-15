from flask import Flask, request, jsonify
from hosp_utils.es_functions import query_elasticsearch_hosp, filtering_hosp
from pharm_utils.es_functions_for_pharmacy import query_elasticsearch_pharmacy
from hosp_utils.recommendation import HospitalRecommender
#사전학습때문에 추가한 두 utils
import torch
from hosp_utils.vae import VAE

from er_utils.apis import *
from er_utils.direction_for_er import *
from er_utils.filtering_for_addr import *
from er_utils.for_redis import *

from utils.direction import calculate_travel_time_and_distance 
from utils.geocode import address_to_coords, coords_to_address
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

from gpt_utils.prompting_gpt import get_medical_info

app = Flask(__name__)

@app.route('/recommend_hospital', methods=['POST'])
def recommend_hospital():
    #전체 시작 시간
    total_start_time = time.time()

    #요청 데이터 수신
    data = request.get_json()
    basic_info = data.get("basic_info")
    health_info = data.get("health_info")
    department = data.get("department", "내과")  #기본값 설정
    suspected_disease = data.get("suspected_disease", None)  #의심 질병
    secondary_hospital = data.get("secondary_hospital", False)
    tertiary_hospital = data.get("tertiary_hospital", False)

    #Geocoding(주소 -> 위도, 경도)
    geocoding_start_time = time.time()

    #data = request.json  #JSON 데이터 파싱
    
    #사용자 실제 현 위치
    user_lat = data.get('lat')
    user_lon = data.get('lon')

    try:
        coords = address_to_coords(basic_info['address'])
        if "error" in coords:
            return jsonify({"error": coords["error"]}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not user_lat or not user_lon:
        user_lat = coords['lat']
        user_lon = coords['lon']

    geocoding_end_time = time.time()
    print(f"Geocoding Time: {geocoding_end_time - geocoding_start_time:.2f} seconds")


    #Elasticsearch 검색
    es_start_time = time.time()
    es_results = query_elasticsearch_hosp(user_lat, user_lon, department, secondary_hospital, tertiary_hospital)
    if "hits" not in es_results or not es_results["hits"]["hits"]:
        return jsonify({"message": "No hospitals found"}), 404
    es_end_time = time.time()
    print(f"Elasticsearch Query Time: {es_end_time - es_start_time:.2f} seconds")

    #필터링된 결과 추출
    filtering_start_time = time.time()
    filtered_hospitals = filtering_hosp(es_results)
    hospital_data = [hospital for hospital in filtered_hospitals]
    df = pd.DataFrame(hospital_data)
    filtering_end_time = time.time()
    print(f"Filtering Time: {filtering_end_time - filtering_start_time:.2f} seconds")


    #병원 이동 소요 시간 계산(멀티 쓰레딩 적용)
    travel_start_time = time.time()
    travel_infos = []
    #병렬 처리
    with ThreadPoolExecutor(max_workers=10) as executor:
        travel_infos = list(
            executor.map(
                lambda row: calculate_travel_time_and_distance(row, user_lat, user_lon),
                df.to_dict("records")
            )
        )
    travel_end_time = time.time()
    print(f"🔹 Total Hospitals: {len(df)}")
    print(f"🔹 Travel Time API Calls: {len(travel_infos)}")
    print(f"Travel Time Calculation: {travel_end_time - travel_start_time:.2f} seconds")

    #DataFrame에 반영
    df['travel_info'] = travel_infos
    
    #대중교통 모드 관련 컬럼 추가
    df["transit_travel_distance_km"] = df['travel_info'].apply(
        lambda x: x.get("transit_travel_distance_km") if x else None
    )
    df["transit_travel_time_h"] = df['travel_info'].apply(
        lambda x: x.get("transit_travel_time_h") if x else None
    )
    df["transit_travel_time_m"] = df['travel_info'].apply(
        lambda x: x.get("transit_travel_time_m") if x else None
    )
    df["transit_travel_time_s"] = df['travel_info'].apply(
        lambda x: x.get("transit_travel_time_s") if x else None
    )

    df.drop(columns=["travel_info"], inplace=True)

    #추천 시스템
    recommend_start_time = time.time()
    recommender = HospitalRecommender()
    user_embedding = recommender.embed_user_profile(basic_info, health_info)

    hospital_embeddings = recommender.embed_hospital_data(df, suspected_disease=suspected_disease)
    

    #사전학습된 VAE 로드
    
    vae = VAE(input_dim=hospital_embeddings.shape[1], hidden_dim=32, latent_dim=16)
    vae.load_state_dict(torch.load("vae_pretrained_model_ld16hd32_v2.pth"))
    vae.eval()  #평가 모드 설정
    
    recommended_hospitals = recommender.recommend_hospitals(
        user_embedding=user_embedding,
        hospital_embeddings=hospital_embeddings,
        hospitals_df=df,
        vae=vae,
        department=department,
        suspected_disease=suspected_disease,
        use_vae=True
    )
    recommend_end_time = time.time()
    print(f"Recommendation System Time: {recommend_end_time - recommend_start_time:.2f} seconds")

    recommended_hospitals["total_travel_time_sec"] = (
    recommended_hospitals["transit_travel_time_h"].fillna(0) * 3600 +
    recommended_hospitals["transit_travel_time_m"].fillna(0) * 60 +
    recommended_hospitals["transit_travel_time_s"].fillna(0)
    )

    #최종 정렬: 이동시간 정렬 후 similarity 정렬
    recommended_hospitals = recommended_hospitals.sort_values(by=["total_travel_time_sec","similarity"], ascending=[True,False])
    recommended_hospitals = recommended_hospitals.drop(columns=["total_travel_time_sec"])
    recommended_hospitals = recommended_hospitals.reset_index(drop=True)

    #전체 종료 시간
    total_end_time = time.time()
    print(f"Total Processing Time: {total_end_time - total_start_time:.2f} seconds")

    #결과 반환
    return jsonify(recommended_hospitals.to_dict(orient="records"))

@app.route('/recommend_pharmacy', methods=['POST'])
def recommend_pharmacy():
    data = request.json  #JSON 데이터 파싱
    user_lat = data.get('lat')
    user_lon = data.get('lon')
    basic_info = data.get("basic_info")

    try:
        coords = address_to_coords(basic_info['address'])
        if "error" in coords:
            return jsonify({"error": coords["error"]}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not user_lat or not user_lon:
        user_lat = coords['lat']
        user_lon = coords['lon']

    #Elasticsearch 쿼리 실행
    es_results = query_elasticsearch_pharmacy(user_lat, user_lon)


    if "hits" in es_results and es_results['hits']['total']['value'] > 0:
        pharmacy_data = [hit['_source'] for hit in es_results['hits']['hits']]
        df = pd.DataFrame(pharmacy_data)

        #열 이름 변경(멀티쓰레딩 전에 처리)
        df.rename(columns={
            'wgs84lat': 'latitude',
            'wgs84lon': 'longitude',
            'dutyaddr': 'address'
        }, inplace=True)


        #약국 이동 소요 시간 계산(멀티 쓰레딩 적용)
        travel_infos = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            travel_infos = list(
                executor.map(
                    lambda row: calculate_travel_time_and_distance(row, user_lat, user_lon),
                    df.to_dict("records")
                )
            )

        #DataFrame에 반영
        df['travel_info'] = travel_infos

        #대중교통 모드 관련 컬럼 추가
        df["transit_travel_distance_km"] = df['travel_info'].apply(
            lambda x: x.get("transit_travel_distance_km") if x else None
        )
        df["transit_travel_time_h"] = df['travel_info'].apply(
            lambda x: x.get("transit_travel_time_h") if x else None
        )
        df["transit_travel_time_m"] = df['travel_info'].apply(
            lambda x: x.get("transit_travel_time_m") if x else None
        )
        df["transit_travel_time_s"] = df['travel_info'].apply(
            lambda x: x.get("transit_travel_time_s") if x else None
        )

        df.drop(columns=["travel_info"], inplace=True)

        #결과 반환
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({"message": "No pharmacies found"}), 404

@app.route('/recommend_er', methods=['POST'])
def recommend_er():
    data = request.json  #JSON 데이터 파싱
    conditions_korean = data.get('conditions', [])  #기본값 빈 리스트

    #설정 파일 로드
    address_filter = AddressFilter()

    #1. 사용자 주소 -> 좌표 변환
    user_lat = data.get('lat')
    user_lon = data.get('lon')
    basic_info = data.get('basic_info', {})
    address = basic_info['address']
    try:
        coords = address_to_coords(address)
        if "error" in coords:
            return jsonify({"error": coords["error"]}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not user_lat or not user_lon:
        #1-1.사용자 좌표가 없는 경우, 주소로부터 좌표를 사용
        user_lat = coords['lat']
        user_lon = coords['lon']
    else:
        try:
            #사용자 좌표와 주소로 변환된 좌표 비교
            converted_address = coords_to_address(user_lat, user_lon)
            if "error" not in converted_address:
                converted_coords = address_to_coords(converted_address['address_name'])

                if converted_coords['lat'] != coords['lat'] or converted_coords['lon'] != coords['lon']:
                    lat_diff = abs(converted_coords['lat'] - coords['lat'])
                    lon_diff = abs(converted_coords['lon'] - coords['lon'])

                    #자그마한 차이일 경우 필터링하지 않고 그대로 사용
                    if lat_diff < 0.00001 and lon_diff < 0.00001:
                        pass  #응급실 추천에서는 data.get('lat'), data.get('lon') 그대로 사용
                    else:
                        #큰 차이가 있는 경우, 좌표를 기준으로 주소를 재설정
                        address = converted_address['address_name']
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    #2. 병원 조건 설정
    stage1, stage2 = address.split()[:2]  #Stage1 = 시도, Stage2 = 시군구

    #3. 병원 조건에 맞는 hpid 수집
    condition_mapping = {
        "조산산모": "MKioskTy8",
        "정신질환자": "MKioskTy9",
        "신생아": "MKioskTy10",
        "중증화상": "MKioskTy11"
    }
    
    conditions = [condition_mapping[cond] for cond in conditions_korean if cond in condition_mapping]

    hpid_list = get_hospitals_by_condition(stage1, stage2, conditions)
    if not hpid_list:
        return jsonify({"message": "No hospitals found for the given conditions"}), 404

    #4. 실시간 병상 정보 조회
    real_time_data = get_real_time_bed_info(stage1, stage2, hpid_list)
    if not real_time_data:
        return jsonify({"message": "No real-time bed information available"}), 404

    #5. 병상 정보 DataFrame 생성
    df = pd.DataFrame(real_time_data)

    #6. enriched_df 생성 및 저장
    enriched_df = address_filter.enrich_filtered_df(df)

    #7. 소요 시간 계산 및 정렬
    enriched_df = calculate_travel_time_and_sort(enriched_df, user_lat, user_lon)
    
    #필요한 열만 선택
    columns_to_return = ["dutyName", "dutyAddr", "dutyTel3", "hvamyn", "is_trauma",
                         "transit_travel_distance_km", "transit_travel_time_h",
                         "transit_travel_time_m", "transit_travel_time_s", "wgs84Lat", "wgs84Lon"]
    
    filtered_df = enriched_df[columns_to_return]

    #결과 반환
    return jsonify(filtered_df.to_dict(orient='records'))

#증상, 언어 -> 병명, 질문&체크리스트
@app.route('/process_symptoms', methods=['POST'])
def process_symptoms():
    try:
        #JSON 데이터 받기
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        language = data.get('language')

        if not symptoms or not language:
            return jsonify({"error": "Both 'symptoms' and 'language' are required"}), 400

        #GPT API 호출
        result = get_medical_info(symptoms, language)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/geocode/address_to_coords', methods=['POST'])
def geocode_address_to_coords():
    """
    주소를 받아 위도와 경도로 변환하여 반환하는 API 엔드포인트
    Request:
    {
        "address": "도로명 주소"
    }
    Response:
    {
        "lat": 위도,
        "lon": 경도
    }
    """
    try:
        data = request.get_json()
        address = data.get('address')

        if not address:
            return jsonify({"error": "'address' is required"}), 400

        coords = address_to_coords(address)

        if "error" in coords:
            return jsonify({"error": coords["error"]}), 400

        return jsonify(coords), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/geocode/coords_to_address', methods=['POST'])
def geocode_coords_to_address():
    """
    위도와 경도를 받아 주소로 변환하여 반환하는 API 엔드포인트
    Request:
    {
        "lat": 위도,
        "lon": 경도
    }
    Response:
    {
        "address_name": "도로명 주소"
    }
    """
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')

        if lat is None or lon is None:
            return jsonify({"error": "Both 'lat' and 'lon' are required"}), 400

        address = coords_to_address(lat, lon)

        if "error" in address:
            return jsonify({"error": address["error"]}), 400

        return jsonify(address), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)