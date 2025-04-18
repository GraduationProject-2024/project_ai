from flask import request, jsonify, Flask
from text_119_utils.stt_tts_translation import transcribe_audio, translate_and_filter_text#, generate_tts

from text_119_utils.s3_utils import upload_to_s3, upload_image_to_s3, download_from_s3_image
import os
from text_119_utils.selenium_test import setup_driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import time
from text_119_utils.en_juso import get_english_address 
from text_119_utils.ai_for_form import *
from text_119_utils.pw_gen import *

from text_119_utils.cleaning import clean_form_value
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

#119 신고용 비밀번호 만들기
@app.route("/reportapi/gen_119_pw", methods=["GET"])
def generate_password_api():
    """비밀번호 생성 API"""
    used_passwords = get_used_passwords()  #MySQL에서 used_passwords 조회
    password = generate_password(used_passwords)
    return jsonify({"password": password})


#웹으로 신고
#postman에서 form-data 형태로 POST
@app.route("/reportapi/fill_form", methods=["POST"])
def fill_form():
    print("===== 프론트에서 받은 요청 =====", flush=True)
    print("form:", request.form, flush=True)
    print("files:", request.files, flush=True)
    print("values:", {key: request.form.get(key) for key in request.form}, flush=True)
    print("files keys:", list(request.files.keys()), flush=True)
    print("=============================", flush=True)
    """프론트엔드에서 받은 값을 입력 필드에 채운 후 버튼 클릭"""
    name = clean_form_value(request.form.get("name", "테스트 이름"))
    phone_number = clean_form_value(request.form.get("number", "01012345678"))
    parts = []
    parts.append(phone_number[:3])
    parts.append(phone_number[3:7])
    parts.append(phone_number[7:])


    #password:6~16 digits.
    password = request.form.get("119_gen_pw", "rSYNshcgPjqd")
    location = clean_form_value(request.form.get("incident_location"))
    print('location:', location, flush=True)
    if not location:
        location = clean_form_value(request.form.get("address"))
        print('첫번째 if문 처리:', location, flush=True)
    if not location:
        location = "서울특별시 용산구 청파로47길 100"
        print("기본 주소로 처리됨", location, flush=True)

    print('최종 location', flush=True)
    print(type(location), flush=True)

    #STT 처리: audio 파일이 있을 경우
    if "audio" in request.files:
        audio_file = request.files["audio"]
        s3_audio_url = upload_to_s3(audio_file, "audio/transcript/")
        transcribed_text, local_audio_path = transcribe_audio(s3_audio_url)
        os.remove(local_audio_path)  #임시 파일 삭제
        content = transcribed_text
    else:
        content = clean_form_value(request.form.get("content", None))
        #content = request.form.get("content", "").strip()

    #content가 없거나 비어 있을 경우 GPT 호출 생략
    if content is None or content.strip().lower() in ("", "null", "none", 'null') or content == 'null':
        #GPT 호출하지 않고 기본값 직접 지정
        content_ko = "신고 내용이 없습니다."
        content_en = "No report content provided."
        processed_content = f"{content_en}({content_ko})"
        default_title_ko = "긴급 신고"
        default_title_en = "Emergency Report"
        default_emergency_type = "Emergency"
    else:
        print('else구문의 content:', content)
        content_ko, content_en = summarize_content(content)
        processed_content = f'{content_en}({content_ko})'
        default_title_ko, default_title_en, default_emergency_type = generate_title_and_type(processed_content)
        
    print("[요약된 content_ko]:", content_ko)
    print("[요약된 content_en]:", content_en)


    #기본값 제목 글자수 제한
    if default_title_ko:
        default_title = f"{default_title_en} ({default_title_ko})"
    else:
        default_title = default_title_en
    if len(default_title) > 100:
        default_title = default_title[:100]
    print("[생성된 title]:", default_title_ko, default_title_en)
    print("default_title:", default_title)
    
    raw_emergency_type = clean_form_value(request.form.get("emergency_type", "Emergency"))

    if (
        raw_emergency_type is None or
        raw_emergency_type == "null" or
        raw_emergency_type.strip().lower() in ("", "null", "none")
    ):
        emergency_type = default_emergency_type
    else:
        emergency_type = raw_emergency_type.strip()

    title = clean_form_value(request.form.get("title")) or default_title


    #이미지 업로드 처리:3개까지 가능 (optional)
    image_urls = []
    for file_key in ["file_1", "file_2", "file_3"]:
        if file_key in request.files:
            image_file = request.files[file_key]
            s3_url = upload_image_to_s3(image_file)
            image_urls.append(s3_url)

    #한국어 주소 -> select 옵션 매핑
    sido_mapping = {
        "서울특별시": "11",
        "부산광역시": "26",
        "대구광역시": "27",
        "인천광역시": "28",
        "광주광역시": "29",
        "대전광역시": "30",
        "울산광역시": "31",
        "세종특별자치시": "36",
        "경기도": "41",
        "강원도": "42",
        "충청북도": "43",
        "충청남도": "44",
        "전라북도": "45",
        "전라남도": "46",
        "경상북도": "47",
        "경상남도": "48",
        "제주특별자치도": "49"
    }

    driver = setup_driver()
    
    try:
        #119 페이지 열기
        driver.get("https://www.119.go.kr/Center119/registEn.do")
        wait = WebDriverWait(driver, 10)
        #이름 필드가 로드될 때까지 대기
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="dsr_name"]')))

        #입력 필드 찾기 및 값 입력
        driver.find_element(By.XPATH, '//*[@id="dsr_name"]').send_keys(name)
        driver.find_element(By.XPATH, '//*[@id="call_tel1"]').send_keys(parts[0])
        driver.find_element(By.XPATH, '//*[@id="call_tel2"]').send_keys(parts[1])
        driver.find_element(By.XPATH, '//*[@id="call_tel3"]').send_keys(parts[2])

        #신고 유형 선택
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="dsrKndCdList"]')))
        select_element = Select(driver.find_element(By.XPATH, '//*[@id="dsrKndCdList"]'))
        for option in select_element.options:
            if option.text.strip().lower() == emergency_type.strip().lower():
                select_element.select_by_visible_text(option.text.strip())
                break
        #제목 입력
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="title"]')))
        driver.find_element(By.XPATH, '//*[@id="title"]').send_keys(title)
        time.sleep(1)
        #시/도 코드 선택
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="sidoCode"]')))
        for region, code in sido_mapping.items():
            if region in location:
                Select(driver.find_element(By.XPATH, '//*[@id="sidoCode"]')).select_by_value(code)
                time.sleep(1) #선택 반영 대기
                break
        
        #주소 영문 변환 및 세부 주소 입력
        eng_location = get_english_address(location)
        print("eng_location:", eng_location)
        if eng_location:
            wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="juso"]')))
            address_input = driver.find_element(By.XPATH, '//*[@id="juso"]')
            address_input.clear()
            address_input.send_keys(eng_location)
        

        #신고 내용 본문
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="contents"]')))
        driver.find_element(By.XPATH, '//*[@id="contents"]').send_keys(processed_content)

        #비밀번호
        driver.find_element(By.XPATH, '//*[@id="userPw"]').send_keys(password)

        #S3 이미지 다운로드 후 파일 업로드
        for idx, s3_url in enumerate(image_urls):
            local_file = download_from_s3_image(s3_url)  #로컬 다운로드
            print(local_file)
            file_input_xpath = f'//*[@id="file_{idx+1}"]'  #file_1, file_2, file_3
            wait.until(EC.presence_of_element_located((By.XPATH, file_input_xpath)))
            driver.find_element(By.XPATH, file_input_xpath).send_keys(local_file)
            print(f'{idx+1}번째 파일 첨부 완료')
            os.remove(local_file)  #업로드 후 삭제

        #버튼 클릭(실 사용에서는 send 버튼으로 변경 필요)
        wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div/div[2]/div[2]/div/nav/ul/li[2]/button')))
        button = driver.find_element(By.XPATH, '/html/body/div[5]/div/div[2]/div[2]/div/nav/ul/li[2]/button')
        button.click()

        return jsonify({"status": "success", "message": "버튼 클릭 완료"})
    
    except TimeoutException as e:
        return jsonify({"status": "error", "message": "요소 로딩 대기 중 시간 초과: " + str(e)})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
    finally:
        print('끝')
        driver.quit()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
    #app.run()