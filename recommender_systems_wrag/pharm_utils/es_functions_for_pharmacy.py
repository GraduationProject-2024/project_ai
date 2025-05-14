from elasticsearch import Elasticsearch
import configparser
config=configparser.ConfigParser()
config.read('keys.config')

es=Elasticsearch(
    hosts=[config['ES_INFO']['host']],
    basic_auth=(config['ES_INFO']['username'], config['ES_INFO']['password']),
    verify_certs=False
)

def query_elasticsearch_pharmacy(user_lat, user_lon):
    """
    Elasticsearch에서 사용자 위치와 가까운 약국 검색.
    """
    query={
        "query": {
            "bool": {
                "filter": [
                    {
                        "geo_distance": {
                            "distance": "100km",  #100km 제한
                            "location": {
                                "lat": user_lat,
                                "lon": user_lon
                            }
                        }
                    }
                ]
            }
        },
        "_source": True,  #모든 필드를 _source로 가져옴
        "sort": [
            {
                "_geo_distance": {
                    "location": {"lat": user_lat, "lon": user_lon},
                    "order": "asc",
                    "unit": "km"
                }
            }
        ],
        "size": 15  #최대 15개 결과 제한
    }

    #Elasticsearch 검색 실행
    es_results=es.search(index="pharmacy_records_v2", body=query)
    return es_results
