from elasticsearch import Elasticsearch
import os
# Elasticsearch 클라이언트 설정
es = Elasticsearch(os.getenv("ES_HOST"))

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
        "sort": [
            {
                "_geo_distance": {
                    "location": {"lat": user_lat, "lon": user_lon},
                    "order": "asc",
                    "unit": "km"
                }
            }
        ]
    }

    # Elasticsearch 검색 실행
    es_results = es.search(index="pharmacy_records_v2", body=query)
    return es_results
