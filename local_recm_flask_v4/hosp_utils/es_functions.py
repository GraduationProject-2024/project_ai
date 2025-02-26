def query_elasticsearch_hosp(user_lat, user_lon, department=None, secondary_hospital=False, tertiary_hospital=False):
    from elasticsearch import Elasticsearch
    import configparser
    config = configparser.ConfigParser()
    config.read('keys.config')
    #Elasticsearch 클라이언트 설정
    es = Elasticsearch(
        hosts=[config['ES_INFO']['host']],
        basic_auth=(config['ES_INFO']['username'], config['ES_INFO']['password']),
        #ca_certs="./local_recm_flask/http_ca.crt",  # 로컬에 저장된 CA 인증서 경로
        verify_certs=False
    )

    #병원 유형 필터 구성
    must_clcdnm = []
    if secondary_hospital:
        must_clcdnm.extend(["병원", "종합병원"])
    if tertiary_hospital:
        must_clcdnm.append("상급종합")

    #Elasticsearch 쿼리 구성
    must_queries = []
    # 특수한 department 케이스 처리
    if department == "치의과":
        dental_departments = [
            "치과", "구강악안면외과", "치과보철과", "치과교정과", "소아치과",
            "치주과", "치과보존과", "구강내과", "영상치의학과", "구강병리과",
            "예방치과", "통합치의학과"
        ]
        must_queries.append({"terms": {"dgsbjt": dental_departments}})
    elif department == "한방과":
        oriental_departments = [
            "한방내과", "한방부인과", "한방소아과", "한방안·이비인후·피부과",
            "한방신경정신과", "침구과", "한방재활의학과", "사상체질과", "한방응급"
        ]
        must_queries.append({"terms": {"dgsbjt": oriental_departments}})
    elif department:
        # 기존 department 처리
        must_queries.append({"match_phrase": {"dgsbjt": department}})

    #메인 쿼리
    query = {
        "_source": True,  #_source 필드 포함
        "query": {
            "bool": {
                "must": must_queries,
                "filter": [
                    {"geo_distance": {"distance": "100km", "location": {"lat": user_lat, "lon": user_lon}}}
                ]
            }
        },
        "sort": [
            {"_geo_distance": {"location": {"lat": user_lat, "lon": user_lon}, "order": "asc", "unit": "km"}}
        ],
        "script_fields": {  # 거리 값을 반환하도록 스크립트 필드 추가
            "es_distance_in_km": {
                "script": {
                    "source": "doc['location'].arcDistance(params.lat, params.lon) / 1000",  # km 거리 반환
                    "params": {"lat": user_lat, "lon": user_lon}
                }
            }
        },
        "size": 50  # 최대 50개 결과 제한
    }

    #병원 유형 필터 추가
    if must_clcdnm:
        query["query"]["bool"]["must"].append({"terms": {"clcdnm": must_clcdnm}})
    
    # Elasticsearch 검색 실행
    response = es.search(
        index="hospital_records_v3",
        body=query
    )
    
    return response

def filtering_hosp(results):
    """
    Elasticsearch 결과 필터링
    """
    filtered_results = []
    for hit in results['hits']['hits']:
        source = hit['_source']

        es_distance_in_km = hit.get("fields", {}).get("es_distance_in_km", [None])[0] #script_fields 값 읽기
        filtered_results.append({
            "id": source.get("id"),
            "name": source.get("yadmnm"),
            "address": source.get("addr"),
            "telephone": source.get("telno"),
            "department": source.get("dgsbjt"),
            "latitude": source.get("ypos"),
            "longitude": source.get("xpos"),
            "es_distance_in_km": es_distance_in_km, 
            "sidocdnm": source.get("sidocdnm"),
            "sggucdnm": source.get("sggucdnm"),
            "emdongnm": source.get("emdongnm"),
            "clcdnm": source.get("clcdnm"),
            "location": source.get("location"),
            "url": source.get("hospurl"),
            "sort_score": hit.get("sort", [None])[0]  # 정렬 기준 추가
        })
    return filtered_results