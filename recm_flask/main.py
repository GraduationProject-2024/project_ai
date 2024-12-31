from flask import Flask, request, jsonify
from recm_flask.hosp_utils.es_functions import query_elasticsearch_hosp, filtering_hosp
from recm_flask.pharm_utils.es_functions_for_pharmacy import query_elasticsearch_pharmacy
from hosp_utils.recommendation import HospitalRecommender

from er_utils.apis import *
from er_utils.direction_for_er import *
from er_utils.filtering_for_addr import *
from er_utils.for_redis import *

from utils.direction import get_travel_time
from utils.geocode import address_to_coords
import pandas as pd
import time

from gpt_utils.prompting_gpt import get_medical_info

app = Flask(__name__)

@app.route('/recommend_hospital', methods=['POST'])
def recommend_hospital():
    # 요청 데이터 수신
    data = request.get_json()
    basic_info = data.get("basic_info")
    health_info = data.get("health_info")
    department = data.get("department", "내과")  # 기본값 설정
    suspected_disease = data.get("suspected_disease", None)  # 의심 질병
    secondary_hospital = data.get("secondary_hospital", False)
    tertiary_hospital = data.get("tertiary_hospital", False)

    # Geocoding (주소 -> 위도, 경도)
    try:
        coords = address_to_coords(basic_info['address'])
        if "error" in coords:
            return jsonify({"error": coords["error"]}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    user_lat = coords['lat']
    user_lon = coords['lon']

    # Elasticsearch 검색
    es_results = query_elasticsearch_hosp(user_lat, user_lon, department, secondary_hospital, tertiary_hospital)
    if "hits" not in es_results or not es_results["hits"]["hits"]:
        return jsonify({"message": "No hospitals found"}), 404

    # 필터링된 결과 추출
    filtered_hospitals = filtering_hosp(es_results)
    hospital_data = [hospital for hospital in filtered_hospitals]
    df = pd.DataFrame(hospital_data)

    # 병원 이동 소요 시간 계산
    travel_times = []
    for _, row in df.iterrows():
        try:
            hospital_lat = row["latitude"]
            hospital_lon = row["longitude"]
            travel_time_sec = get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon)
            travel_times.append(travel_time_sec or 0)
            time.sleep(0.3)  # API 호출 제한 준수
        except Exception as e:
            travel_times.append(None)

    # DataFrame에 소요 시간 추가
    df['travel_time'] = travel_times  # 원본 소요 시간 (초 단위)
    df['travel_time_sec'] = df['travel_time'].apply(lambda x: x % 60 if x is not None else None)  # 초 단위만 저장
    df['travel_time_h'] = df['travel_time'] // 3600
    df['travel_time_min'] = (df['travel_time'] % 3600) // 60

    # 추천 시스템
    recommender = HospitalRecommender()
    user_embedding = recommender.embed_user_profile(basic_info, health_info)
    hospital_embeddings = recommender.embed_hospital_data(df, suspected_disease=suspected_disease)
    vae = recommender.train_vae(hospital_embeddings, input_dim=hospital_embeddings.shape[1], latent_dim=64, hidden_dim=128)

    recommended_hospitals = recommender.recommend_hospitals(
        vae=vae,
        user_embedding=user_embedding,
        hospital_embeddings=hospital_embeddings,
        hospitals_df=df,
        department=department,
        suspected_disease=suspected_disease,
        use_vae=True
    )

    # 결과 반환
    return jsonify(recommended_hospitals.to_dict(orient="records"))

@app.route('/recommend_pharmacy', methods=['POST'])
def recommend_pharmacy():
    data = request.json  # JSON 데이터 파싱
    user_lat = data.get('lat')
    user_lon = data.get('lon')

    if not user_lat or not user_lon:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    # Elasticsearch 쿼리 실행
    es_results = query_elasticsearch_pharmacy(user_lat, user_lon)

    if "hits" in es_results and es_results['hits']['total']['value'] > 0:
        pharmacy_data = [hit['_source'] for hit in es_results['hits']['hits']]
        df = pd.DataFrame(pharmacy_data)

        # 소요 시간 계산
        df['travel_time_sec'] = df.apply(
            lambda row: get_travel_time(user_lat, user_lon, row['wgs84lat'], row['wgs84lon']),
            axis=1
        )
        df['travel_time_h'] = df['travel_time_sec'] // 3600
        df['travel_time_min'] = (df['travel_time_sec'] % 3600) // 60

        # 결과 반환
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({"message": "No pharmacies found"}), 404

@app.route('/recommend_er', methods=['POST'])
def recommend_er():
    data = request.json  # JSON 데이터 파싱
    address = data.get('address')
    conditions_korean = data.get('conditions', [])  # 기본값 빈 리스트

    if not address:
        return jsonify({"error": "Address is required"}), 400

    # 설정 파일 로드
    config_path = 'C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config'
    address_filter = AddressFilter(config_path)

    # 1. 사용자 주소 -> 좌표 변환
    user_coords = address_to_coords(address)

    if "error" in user_coords:
        return jsonify({"error": user_coords['error']}), 500

    user_lat = user_coords["lat"]
    user_lon = user_coords["lon"]

    # 2. 병원 조건 설정
    stage1, stage2 = address.split()[:2]  # Stage1 = 시도, Stage2 = 시군구

    # 3. 병원 조건에 맞는 hpid 수집
    condition_mapping = {
        "조산산모": "MKioskTy8",
        "정신질환자": "MKioskTy9",
        "신생아": "MKioskTy10",
        "중증화상": "MKioskTy11"
    }
    if not conditions_korean:
        conditions = []  # 빈 리스트로 설정
    else:
        conditions = [condition_mapping[cond] for cond in conditions_korean if cond in condition_mapping]

    hpid_list = get_hospitals_by_condition(stage1, stage2, conditions)
    if not hpid_list:
        return jsonify({"message": "No hospitals found for the given conditions"}), 404

    # 4. 실시간 병상 정보 조회
    real_time_data = get_real_time_bed_info(stage1, stage2, hpid_list)
    if not real_time_data:
        return jsonify({"message": "No real-time bed information available"}), 404

    # 5. 병상 정보 DataFrame 생성
    df = pd.DataFrame(real_time_data)

    # 6. enriched_df 생성 및 저장
    enriched_df = address_filter.enrich_filtered_df(df)

    # 7. 소요 시간 계산 및 정렬
    enriched_df = calculate_travel_time_and_sort(enriched_df, user_lat, user_lon)

    # 필요한 열만 선택
    columns_to_return = [
        'dutyName', 'dutyAddr', 'dutyTel3', 'distance_km',
        'travel_time_h', 'travel_time_m', 'travel_time_s', 'congestion', 'hvamyn', 'is_trauma'
    ]
    filtered_df = enriched_df[columns_to_return]

    # 결과 반환
    return jsonify(filtered_df.to_dict(orient='records'))

#증상, 언어 -> 병명, 질문&체크리스트
@app.route('/process_symptoms', methods=['POST'])
def process_symptoms():
    try:
        # JSON 데이터 받기
        data = request.json
        symptoms = data.get('symptoms')
        language = data.get('language')

        if not symptoms or not language:
            return jsonify({"error": "Both 'symptoms' and 'language' are required"}), 400

        # GPT API 호출
        result = get_medical_info(symptoms, language)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)