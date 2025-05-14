import requests
import configparser
config = configparser.ConfigParser()
config.read('keys.config')

#주소 검색 API 정보
API_URL = "https://business.juso.go.kr/addrlink/addrEngApi.do"
CONFIRM_KEY = config['API_KEYS']['en_juso_api_key']  #발급받은 승인키 입력

def get_english_address(korean_address):
    #한국어 주소 -> 영문 주소소
    params = {
        'confmKey': CONFIRM_KEY,
        'currentPage': 1,
        'countPerPage': 1,
        'keyword': korean_address,
        'resultType': 'json'
    }

    response = requests.get(API_URL, params=params)

    if response.status_code == 200:
        try:
            data = response.json()
            
            #API 응답 검증
            if data['results']['common']['errorCode'] == '0':  #정상 응답
                juso_list = data['results']['juso']
                
                if juso_list:
                    road_addr = juso_list[0].get('roadAddr', None)  #영문 도로명주소
                    jibun_addr = juso_list[0].get('jibunAddr', None)  #영문 지번주소
                    
                    #도로명 주소 우선 반환, 없으면 지번주소 반환
                    return road_addr if road_addr else jibun_addr
                
            else:
                print(f"API 오류: {data['results']['common']['errorMessage']}")
        
        except Exception as e:
            print(f"JSON 파싱 오류: {e}")
    
    else:
        print(f"요청 실패: HTTP 상태 코드 {response.status_code}")

    return None