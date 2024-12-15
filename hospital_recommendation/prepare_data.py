#CSV 파일에서 진료과목 코드 매핑
def get_department_code(department_name):
    import pandas as pd

    df = pd.read_csv('./컬럼정보_코드.csv', encoding='cp949')
    #'명칭' 열을 기준으로 진료과목 코드 찾기
    filtered_row = df[df['명칭'] == department_name]
    if not filtered_row.empty:
        return filtered_row['코드'].values[0] #첫 번째 일치 값 반환
    return None

#병원 데이터 저장
def save_to_mysql(data):
    import configparser

    #ConfigParser 초기화
    config = configparser.ConfigParser()

    #keys.config 파일 읽기
    config.read('../keys.config')

    import pymysql
    connection = pymysql.connect(host = config['DB_INFO']['host'],
                        port = 3306,
                        user = config['DB_INFO']['id'],
                        passwd = config['DB_INFO']['password'],
                        db = config['DB_INFO']['db'],
                        charset = 'utf8') # 한글 깨짐 방지
        
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    
    #새 데이터 삽입
    insert_query = """
        INSERT INTO hospital_data (addr, yadmNm, clCdNm, sidoCdNm, sgguCdNm, emdongNm, xPos, yPos, telno, hospUrl, dgsbjt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    cursor.executemany(insert_query, data)
    connection.commit()
    cursor.close()
    connection.close()

#API 호출 및 데이터 수집
def fetch_and_store_data(department_name, servicekey, pageNo):
    from pprint import pprint #dict 가독성 좋게 출력
    import xmltodict
    import requests

    #진료과 이름을 진료과목 코드로 변환
    department_code = get_department_code(department_name)
    
    #API 요청 구성
    params = {
        'ServiceKey': servicekey,
        'dgsbjtCd': department_code,
        'numOfRows': 1000,
        'pageNo': pageNo
    }
    response = requests.get('http://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList', params=params)
    
    #API 요청 결과 파싱
    data_to_insert = []
    if response.status_code == 200:
        root = xmltodict.parse(response.content)  # XML을 JSON으로 변환
        items = root['response']['body']['items']['item']
        for item in items:
            data_to_insert.append((
                item['addr'],
                item['yadmNm'],
                item.get('clCdNm', None),
                item['sidoCdNm'],
                item.get('sgguCdNm', None),
                item.get('emdongNm', None),
                item.get('XPos', None),
                item.get('YPos', None),
                item.get('telno', None),
                item.get('hospUrl', None), 
                department_name
            ))
    pprint(data_to_insert)
    #MySQL에 데이터 저장
    save_to_mysql(data_to_insert)
    print('저장 완료')