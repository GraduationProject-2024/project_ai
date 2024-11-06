'''open api의 데이터 형식은 xml'''
import mysql.connector
import requests
import pandas as pd

# MySQL DB 연결
mysql_config = {
    'host':"test241104.cp28wwi2825k.us-east-1.rds.amazonaws.com",
    'user':"jinaen",
    'password':"glaktlsshtmzp11",
    'database':"hospitals"
}

# Excel 파일을 로드하고 진료과목 코드 매핑
def get_department_code(department_name):
    df = pd.read_excel('./컬럼정보_코드.xlsx', sheet_name="Sheet1")

    #'명칭'&'코드'열을 기준으로 진료과목 코드를 찾기
    filtered_row = df[df['명칭'] == department_name]
    if not filtered_row.empty:
        return filtered_row['코드'].values[0]  # 첫 번째 일치 값 반환
    return None

# MySQL에 병원 데이터 저장
def save_to_mysql(data):
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    
    # 기존 데이터 삭제
    cursor.execute("DELETE FROM hospital_data")
    connection.commit()
    
    # 새 데이터 삽입
    insert_query = """
        INSERT INTO hospital_data (addr, yadmNm, sidoCd, sgguCd, xPos, yPos, dgsgbjtCd)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(insert_query, data)
    connection.commit()
    cursor.close()
    connection.close()

api_endpoint = 'http://apis.data.go.kr/B551182/hospInfoServicev2'

# API 호출 및 데이터 수집
def fetch_and_store_data(latitude, longitude, department_name):
    #진료과 이름을 진료과목 코드로 변환
    department_code = get_department_code(department_name)
    
    #API 요청 구성
    params = {
        'ServiceKey': 'SUXfVt013L4slaFREhvcXhfFzEULWvfmkVtMcMwBYUEgHWOF2x8X90hz/RAYNo8ODd0Y5RDDpvFIr1agV426WQ==',
        'dgsgbjtCd': department_code,
        'radius': 5000,
        'numOfRows': 5,
        'pageNo': 5
    }
    response = requests.get(api_endpoint, params=params)
    
    # 4. API 요청 결과 파싱
    data_to_insert = []
    if response.status_code == 200:
        root = response.json()  # XML을 JSON으로 변환한 형태를 가정
        for item in root['body']['items']['item']:
            data_to_insert.append((
                item['addr'],
                item['yadmNm'],
                item['sidoCd'],
                item.get('sgguCd', None),
                item['XPos'],
                item['YPos'],
                department_name
            ))
    
    # 5. MySQL에 데이터 저장
    save_to_mysql(data_to_insert)



from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def index_data_from_mysql_to_es():
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()

    cursor.execute("SELECT addr, yadmNm, xPos, yPos, dgsgbjtCd FROM hospitals")
    records = cursor.fetchall()
    
    for record in records:
        addr, yadmNm, xPos, yPos, dgsgbjtCd = record
        doc = {
            "addr": addr,
            "yadmNm": yadmNm,
            "location": {
                "lat": float(yPos),
                "lon": float(xPos)
            },
            "dgsgbjtCd": dgsgbjtCd
        }
        es.index(index="medical_records", document=doc)
        connection.commit()
    
    cursor.close()
    connection.close()

index_data_from_mysql_to_es()

def query_elasticsearch_for_nearby_hospitals(latitude, longitude, department_code):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"dgsgbjtCd": department_code}}
                ]
            }
        },
        "sort": [
            {
                "_geo_distance": {
                    "location": {
                        "lat": latitude,
                        "lon": longitude
                    },
                    "order": "asc",
                    "unit": "km"
                }
            }
        ],
        "_source": ["addr", "yadmNm"]
    }
    # 데이터 검색
    results = es.search(index="medical_records", body=query)
    data = [hit['_source'] for hit in results['hits']['hits']]
    
    # 인덱스 데이터 삭제
    es.delete_by_query(index="medical_records", body={"query": {"match_all": {}}})
    
    return data

def get_additional_info_from_mysql(hospital_name):
    connection = mysql.connector.connect(**mysql_config)
    cursor = connection.cursor()
    
    cursor.execute("SELECT * FROM hospitals WHERE yadmNm = %s", (hospital_name,))
    return cursor.fetchall()