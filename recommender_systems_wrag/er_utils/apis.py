def extract_stage_from_address(address):
    #도로명 주소에서 시도(STAGE1)와 시군구(STAGE2)를 추출
    try:
        #주소를 공백으로 분리
        parts=address.split()
        if len(parts) < 2:
            raise ValueError("주소 형식이 잘못되었습니다. 최소 시도와 시군구가 필요합니다.")
        
        #시도와 시군구 추출
        stage1=parts[0]  #시도
        stage2=parts[1]  #시군구

        return stage1, stage2
    except Exception as e:
        print(f"주소 파싱 오류: {e}")
        return None, None
    

#API 호출 함수
def call_api(url, params):
    import requests

    try:
        response=requests.get(url, params=params)
        response.raise_for_status()
        return response.text  #XML 응답
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return None


def get_hospitals_by_condition(stage1, stage2, conditions):
    #중증질환 조건에 맞는 병원의 hpid를 Redis 캐싱을 통해 필터링

    import math
    import xml.etree.ElementTree as ET
    import json
    import redis
    from er_utils.for_redis import get_redis_client
    import configparser    
    config=configparser.ConfigParser()
    config.read('keys.config')

    
    #Redis 캐싱 키 생성
    redis_key=f"hospitals:{stage1}:{stage2}:{','.join(conditions) if conditions else 'all'}"

    #Redis에서 데이터 조회
    redis_client=get_redis_client()
    cached_data=redis_client.get(redis_key)

    if cached_data:
        print("Redis 캐시에서 병원 데이터 로드")
        return json.loads(cached_data)  #Redis에서 JSON 디코딩
    
    #Redis에 데이터가 없으면 API 호출
    url="http://apis.data.go.kr/B552657/ErmctInfoInqireService/getSrsillDissAceptncPosblInfoInqire"
    params={
        "STAGE1": stage1,
        "STAGE2": stage2,
        "pageNo": 1,
        "numOfRows": 100,
        "serviceKey": config['API_KEYS']['public_portal_api_key']
        }
    hpid_list=[]

    #첫 호출로 totalCount 확인
    data=call_api(url, params)
    if not data:
        return []

    root=ET.fromstring(data)
    total_count=int(root.find("body/totalCount").text)
    total_pages=math.ceil(total_count / params["numOfRows"])

    #모든 페이지 데이터 수집
    for page in range(1, total_pages + 1):
        params["pageNo"]=page
        page_data=call_api(url, params)
        if not page_data:
            continue
        page_root=ET.fromstring(page_data)
        items=page_root.findall(".//item")

        for item in items:
            #OR 조건으로 병원 필터링
            if not conditions or any(
                item.find(cond) is not None and item.find(cond).text.strip() == "Y"
                for cond in conditions
            ):
                hpid=item.find("hpid").text
                hpid_list.append(hpid)
    
    #결과를 Redis에 저장
    try:
        redis_client.setex(redis_key, 300, json.dumps(hpid_list))
    except redis.exceptions.RedisError as e:
        print(f"Redis 데이터 저장 오류: {e}")
        
    return hpid_list


def get_real_time_bed_info(stage1, stage2, hpid_list):
    #응급실 실시간 데이터를 조회하고, Redis 캐싱
    from er_utils.for_redis import get_redis_client
    import json
    import xml.etree.ElementTree as ET
    import configparser
    
    config=configparser.ConfigParser()
    config.read('keys.config')
    

    redis_client=get_redis_client()
    url="http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"
    params={
        "STAGE1": stage1,
        "STAGE2": stage2,
        "pageNo": 1,
        "numOfRows": 100,
        "serviceKey": config['API_KEYS']['public_portal_api_key']  
    }
    result=[]

    #Redis에서 데이터 확인 및 수집
    for hpid in hpid_list:
        redis_key=f"real_time_bed_info:{hpid}"  #Redis 키
        print(f"생성된 Redis 키: {redis_key}")
        
        cached_data=redis_client.get(redis_key)

        if cached_data:
            print(f"Redis에서 {hpid} 데이터 로드")
            result.append(json.loads(cached_data))  #캐시된 데이터를 추가
        else:
            print(f"Redis 캐시 없음, {hpid} 데이터 API 호출 진행")
            #첫 호출로 totalCount 확인
            data=call_api(url, params)
            if not data:
                continue
            
            root=ET.fromstring(data)
            items=root.findall(".//item")
            for item in items:
                if item.find("hpid").text == hpid:  #해당 병원의 데이터만 처리
                    hospital_data={child.tag: child.text for child in item} #모든 태그 그대로 매핑

                    #Redis에 데이터 저장(5분 TTL 설정)
                    redis_client.setex(redis_key, 300, json.dumps(hospital_data))
                    result.append(hospital_data)
    
    return result
