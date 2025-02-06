from elasticsearch import Elasticsearch
import os
#Elasticsearch 클라이언트 설정
es = Elasticsearch(
    hosts=[os.getenv("ES_HOST")],
    basic_auth=(os.getenv("ES_ID"), os.getenv("ES_PW")),
    #ca_certs="./local_recm_flask/http_ca.crt",  # 로컬에 저장된 CA 인증서 경로
    verify_certs=False
)

def query_elasticsearch_pharmacy(user_lat, user_lon):
    """
    Elasticsearch에서 사용자 위치와 가까운 약국 검색.
    """
    query = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "geo_distance": {
                            "distance": "100km",  # 100km 제한
                            "location": {
                                "lat": user_lat,
                                "lon": user_lon
                            }
                        }
                    }
                ]
            }
        },
        "_source": True,  # 모든 필드를 _source로 가져옴
        "sort": [
            {
                "_geo_distance": {
                    "location": {"lat": user_lat, "lon": user_lon},
                    "order": "asc",
                    "unit": "km"
                }
            }
        ],
        "size": 50  # 최대 50개 결과 제한
    }

    # Elasticsearch 검색 실행
    es_results = es.search(index="pharmacy_records_v2", body=query)
    #print(es_results)
    return es_results