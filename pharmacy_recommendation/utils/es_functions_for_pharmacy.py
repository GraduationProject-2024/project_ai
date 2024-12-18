from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 설정
es = Elasticsearch("http://localhost:9200")

def query_pharmacy_elasticsearch(user_lat, user_lon):
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
