import requests
import pandas as pd
import xmltodict


# CSV 파일에서 진료과목 코드 매핑
def get_department_code(department_name):
    df = pd.read_csv('./컬럼정보_코드.csv', encoding='cp949')
    # '명칭' 열을 기준으로 진료과목 코드 찾기
    filtered_row = df[df['명칭'] == department_name]
    if not filtered_row.empty:
        return filtered_row['코드'].values[0]  # 첫 번째 일치 값 반환
    return None

# MySQL에 병원 데이터 저장
def save_to_mysql(data):
    import pymysql
    connection = pymysql.connect(host = 'test241104.cp28wwi2825k.us-east-1.rds.amazonaws.com',
                        port = 3306,
                        user = 'jinaen',
                        passwd = 'glaktlsshtmzp11',
                        db = 'hospitals',
                        charset = 'utf8') # 한글 깨짐 방지
        
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
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

api_endpoint = 'http://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList'

# API 호출 및 데이터 수집
def fetch_and_store_data(department_name):
    from pprint import pprint #dict 가독성 좋게 출력
    #진료과 이름을 진료과목 코드로 변환
    department_code = get_department_code(department_name)
    
    #API 요청 구성
    params = {
        'ServiceKey': 'SUXfVt013L4slaFREhvcXhfFzEULWvfmkVtMcMwBYUEgHWOF2x8X90hz/RAYNo8ODd0Y5RDDpvFIr1agV426WQ==',
        'dgsbjtCd': department_code,
        'radius': 5000,
        'numOfRows': 5,
        'pageNo': 1
    }
    response = requests.get(api_endpoint, params=params)
    #print(response) #<Response [200]>
    
    # 4. API 요청 결과 파싱
    data_to_insert = []
    if response.status_code == 200:
        root = xmltodict.parse(response.content)  # XML을 JSON으로 변환
        items = root['response']['body']['items']['item']
        for item in items:
            data_to_insert.append((
                item['addr'],
                item['yadmNm'],
                item['sidoCd'],
                item.get('sgguCd', None),
                item['XPos'],
                item['YPos'],
                department_code
            ))
    pprint(data_to_insert)

    # 5. MySQL에 데이터 저장
    save_to_mysql(data_to_insert)
fetch_and_store_data('내과')
