def query_elasticsearch(user_lat, user_lon, department=None, secondary_hospital=False, tertiary_hospital=False):
    from elasticsearch import Elasticsearch
    import configparser
    #Scroll API를 사용하여 Elasticsearch에서 사용자 위치와 선택적인 진료과로 병원 검색
    #ConfigParser 초기화
    config = configparser.ConfigParser()
    #keys.config 파일 읽기
    config.read('C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config')
    #Elasticsearch 클라이언트 설정
    es = Elasticsearch(config['ES_INFO']['host'])

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

    # if department:
        # must_queries.append({"match_phrase": {"dgsbjt": department}})
    
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
            "distance_in_m": {
                "script": {
                    "source": "doc['location'].arcDistance(params.lat, params.lon)",  # 미터 단위 거리 반환
                    "params": {"lat": user_lat, "lon": user_lon}
                }
            }
        }
    }

    #병원 유형 필터 추가
    if must_clcdnm:
        query["query"]["bool"]["must"].append({"terms": {"clcdnm": must_clcdnm}})
    
    #Scroll API 사용
    scroll_time = "2m"  # Scroll 컨텍스트 유지 시간
    batch_size = 1000  # 한 번에 가져올 문서 수
    results = []

    #Scroll 초기화
    response = es.search(
        index="hospital_records",
        body=query,
        scroll=scroll_time,
        size=batch_size
    )
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    results.extend(hits)

    #Scroll 반복
    while True:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        if not hits:
            break
        results.extend(hits)

    #Scroll 컨텍스트 해제
    es.clear_scroll(scroll_id=scroll_id)

    #return results
    return {'hits': {'hits': results}}

def filtering(results):
    """
    Elasticsearch 결과 필터링
    """
    filtered_results = []
    for hit in results['hits']['hits']:
        source = hit['_source']

        distance_in_m = hit.get("fields", {}).get("distance_in_m", [None])[0] #script_fields 값 읽기
        filtered_results.append({
            "id": source.get("id"),
            "name": source.get("yadmnm"),
            "address": source.get("addr"),
            "telephone": source.get("telno"),
            "department": source.get("dgsbjt"),
            "latitude": source.get("ypos"),
            "longitude": source.get("xpos"),
            "distance_in_m": distance_in_m, 
            "sidocdnm": source.get("sidocdnm"),
            "sggucdnm": source.get("sggucdnm"),
            "emdongnm": source.get("emdongnm"),
            "clcdnm": source.get("clcdnm"),
            "location": source.get("location"),
            "url": source.get("hospurl"),
            "sort_score": hit.get("sort", [None])[0]  # 정렬 기준 추가
        })
    return filtered_results