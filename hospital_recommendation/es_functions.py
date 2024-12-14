# def query_elasticsearch(user_lat, user_lon, department=None, secondary_hospital=False, tertiary_hospital=False):
#     from elasticsearch import Elasticsearch
#     # Elasticsearch 클라이언트 설정
#     es = Elasticsearch("http://localhost:9200")
    
#     """
#     Elasticsearch에서 사용자 위치와 선택적인 진료과 및 의심 질병으로 병원 검색
#     """
#     must_clcdnm = []
#     if secondary_hospital:
#         must_clcdnm.extend(["병원", "종합병원"])
#     if tertiary_hospital:
#         must_clcdnm.append("상급종합")

#     must_queries = []

#     # 진료과 필터 추가 (선택적)
#     if department:
#         must_queries.append({"match_phrase": {"dgsbjt": department}})

#     query = {
#         "query": {
#             "bool": {
#                 "must": must_queries,
#                 "filter": [
#                     {"geo_distance": {"distance": "500km", "location": {"lat": user_lat, "lon": user_lon}}}
#                 ]
#             }
#         },
#         "sort": [
#             {"_geo_distance": {"location": {"lat": user_lat, "lon": user_lon}, "order": "asc", "unit": "km"}}
#         ]
#     }

#     # 병원 유형 필터 추가
#     if must_clcdnm:
#         query["query"]["bool"]["must"].append({"terms": {"clcdnm": must_clcdnm}})

#     return es.search(index="hospital_records", body=query)

def query_elasticsearch(user_lat, user_lon, department=None, secondary_hospital=False, tertiary_hospital=False):
    from elasticsearch import Elasticsearch
    import configparser
    """
    Scroll API를 사용하여 Elasticsearch에서 사용자 위치와 선택적인 진료과로 병원 검색
    """
    # ConfigParser 초기화
    config = configparser.ConfigParser()
    # keys.config 파일 읽기
    config.read('../keys.config')
    # Elasticsearch 클라이언트 설정
    es = Elasticsearch(config['ES_INFO']['host'])

    # 병원 유형 필터 구성
    must_clcdnm = []
    if secondary_hospital:
        must_clcdnm.extend(["병원", "종합병원"])
    if tertiary_hospital:
        must_clcdnm.append("상급종합")

    # Elasticsearch 쿼리 구성
    must_queries = []
    if department:
        must_queries.append({"match_phrase": {"dgsbjt": department}})
    
    # 메인 쿼리
    query = {
        "query": {
            "bool": {
                "must": must_queries,
                "filter": [
                    {"geo_distance": {"distance": "500km", "location": {"lat": user_lat, "lon": user_lon}}}
                ]
            }
        },
        "sort": [
            {"_geo_distance": {"location": {"lat": user_lat, "lon": user_lon}, "order": "asc", "unit": "km"}}
        ]
    }

    # 병원 유형 필터 추가
    if must_clcdnm:
        query["query"]["bool"]["must"].append({"terms": {"clcdnm": must_clcdnm}})
    
    # Scroll API 사용
    scroll_time = "2m"  # Scroll 컨텍스트 유지 시간
    batch_size = 10000  # 한 번에 가져올 문서 수
    results = []

    # Scroll 초기화
    response = es.search(
        index="hospital_records",
        body=query,
        scroll=scroll_time,
        size=batch_size
    )
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    results.extend(hits)

    # Scroll 반복
    while True:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        if not hits:
            break
        results.extend(hits)

    # Scroll 컨텍스트 해제
    es.clear_scroll(scroll_id=scroll_id)

    # return results
    return {'hits': {'hits': results}}


# def filtering(results):
#     """
#     Elasticsearch 결과 필터링
#     """
#     filtered_results = []
#     # for hit in results['hits']['hits']:
#     for hit in results:
#         # 데이터의 주요 정보 출력
#         filtered_results.append(hit)
#     return filtered_results

def filtering(results):
    """
    Elasticsearch 결과 필터링
    """
    filtered_results = []
    for hit in results['hits']['hits']:
        source = hit['_source']
        filtered_results.append({
            "id": source.get("id"),
            "name": source.get("yadmnm"),
            "address": source.get("addr"),
            "telephone": source.get("telno"),
            "department": source.get("dgsbjt"),
            "latitude": source.get("ypos"),
            "longitude": source.get("xpos"),
            "sort_score": hit.get("sort", [None])[0]  # 정렬 기준 추가
        })
    return filtered_results



# 네이버 길찾기 api 사용 -> 소요 시간 측정
def get_travel_time(user_lat, user_lon, hospital_lat, hospital_lon):
    import requests
    import configparser

    # ConfigParser 초기화
    config = configparser.ConfigParser()

    # keys.config 파일 읽기
    config.read('../keys.config')

    """
    네이버 지도 Directions API를 사용하여 사용자 위치와 병원 간의 자동차 이동 소요시간을 반환합니다.

    Parameters:
    - user_lat (float): 사용자 위도
    - user_lon (float): 사용자 경도
    - hospital_lat (float): 병원 위도
    - hospital_lon (float): 병원 경도

    Returns:
    - int: 이동 소요시간 (초 단위)
    """

    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": config['API_KEYS']['naver_api_key_id'],
        "X-NCP-APIGW-API-KEY": config['API_KEYS']['naver_api_key']
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
            # 총 소요시간을 초 단위로 반환
            return data['route']['trafast'][0]['summary']['duration'] / 1000
        else:
            print(f"API 응답 오류: {data['message']}")
            return None
    else:
        print(f"HTTP 요청 오류: {response.status_code}")
        return None