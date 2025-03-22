#-*- coding: utf-8 -*-

import openai
import json
import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

def get_medical_info(symptoms, language):
    """
    사용자의 증상 정보를 기반으로 GPT를 호출하여 진료과, 의심되는 질병 정보, 의사에게 할 질문 리스트를 JSON으로 반환
    
    Args:
        symptoms (dict): 사용자의 증상 정보. 다음과 같은 키를 포함:
            - 'macro_body_part': 거시적 신체 부위
            - 'micro_body_part': 미시적 신체 부위
            - 'symptom_details': 증상 정보 (빈도, 강도, 고통의 지속시간, 증상발현 후 지난 시간)
            - 'additional_info': 기타사항 (S3 URL 등)
        language (str): 사용자 언어 코드 (e.g., 'en', 'ko', 'vi', 'zh', 'zh-Hant', 'mn')
    
    Returns:
        dict: 진료과, 의심되는 질병, 의사에게 할 질문 리스트가 포함된 JSON 객체
    """
    try:
        #증상 정보를 문자열로 변환
        symptom_descriptions = []
        for symptom in symptoms:
            symptom_descriptions.append(
                f"Macro body part: {symptom.get('macro_body_part', 'N/A')}, "
                f"Micro body part: {symptom.get('micro_body_part', 'N/A')}, "
                f"Details: {symptom.get('symptom_details', 'N/A')}, "
                f"Additional info: {symptom.get('additional_info', 'N/A')}"
            )

        #모든 증상 정보를 하나의 설명으로 결합
        combined_description = " | ".join(symptom_descriptions)

        
        #프롬프트 설정
        prompt = (
            "You are a multilingual medical assistant specializing in professional medical terminology. Your role is to provide accurate translations of medical-related information for individuals who need assistance navigating healthcare systems. Specifically, you help foreigners living in Korea and Koreans living abroad by translating medical information and terms in a way that is both culturally and linguistically appropriate. Ensure that all translations use formal medical terminology and avoid colloquial or overly simplified language."
            "When translating symptoms into Korean, ensure the following translations are used:\n"
            "- 'swelling' should be translated as '붓기' (not '부기').\n"
            "- there is no '식도통'. do not use words not in the korean dictionary.\n"
            "Below is a list of valid medical departments with accurate names in Korean(English, Mongolian, Vietnamese, Chinese, Chinese-Hant):"
            "- 가정의학과('Family Medicine', 'Гэр Бүлийн Анагаах Ухаан', 'y học gia đình', '家庭医学科', '家庭醫學')\n"
            "- 내과('Internal Medicine', 'Дотоод Анагаах Ухаан', 'khoa nội, bệnh viện nội khoa', '内科', '內科')\n"
            "- 마취통증의학과('Anaesthesiology', 'Анестезиологи', 'khoa chứng đau gây mê', '麻醉疼痛医学科', '麻醉痛医学科')\n"
            "- 비뇨의학과('Urology', 'Урологи', 'khoa tiết niệu', '泌尿医学系', '泌尿学系')\n"
            "- 산부인과('Obstetrics and Gynecology', 'Эх барих, эмэгтэйчүүдийн', 'khoa phụ sản, bệnh viện phụ sản', '妇产科', '婦產科')\n"
            "- 성형외과('Plastic&Reconstructive Surgery', 'Хуванцар Ба Сэргээн Засах Мэс Засал', 'Phẫu thuật tạo hình và tái tạo', '整形及重建外科', '整形及重建外科')\n"
            "- 소아청소년과('Pediatrics','хүүхдийн эмч', 'khoa nhi', '儿童青少年科', '小儿青少年科')\n"
            "- 신경과('Neurology', 'Мэдрэл судлал', 'Thần kinh học', '神经科', '神经科')\n"
            "- 신경외과('Neurological Surgery', 'Мэдрэлийн Мэс Засал', 'khoa ngoại thần kinh, bệnh viện ngoại khoa', '神经外科', '神经外科')\n"
            "- 심장혈관흉부외과('Thoracic Surgery', 'Цээжний Мэс Засал', 'khoa ngoại khoa tim mạch', '心血管胸外科', '心血管胸外科')\n"
            "- 안과('Ophthalmology', 'нүдний эмч', 'nhãn khoa, bệnh viện mắt', '眼科', '眼科')\n"
            "- 영상의학과('Imaging Radiology', 'Дүрслэл Радиологи', 'ngành X-quang', '影像医学科', '影像放射學')\n"
            "- 예방의학과('Preventive Medicine', 'Урьдчилан Сэргийлэх Эм', 'Y học dự phòng', '预防医学科', '預防醫學')\n"
            "- 외과('General Surgery', 'Ерөнхий Мэс Засал', 'khoa ngoại, bệnh viện ngoại khoa', '外科', '一般外科')\n"
            "- 이비인후과('Otolaryngology', 'Чих хамар хоолой судлал', 'khoa tai mũi họng, bệnh viện tai mũi họng', '耳鼻喉科', '耳鼻喉科')\n"
            "- 재활의학과('Rehabilitation Medicine', 'Нөхөн Сэргээх Эм', 'thuốc phục hồi chức năng', '康复医学系', '康复医法系')\n"
            "- 정신건강의학과('Psychiatry', 'Сэтгэцийн эмгэг', 'Tâm thần học', '心理健康医学系', '精神健康医学系')\n"
            "- 정형외과('Orthopedic Surgery', 'Ортопедийн Мэс Засал', 'khoa ngoại chỉnh hình, bệnh viện chấn thương chỉnh hình', '骨科手术', '骨科手術')\n"
            "- 치의과('Dentistry', 'Шүдний эмч', 'nha khoa, bệnh viện nha khoa', '牙科', '牙科')\n"
            "- 피부과('Dermatology', 'Арьс судлал', 'khoa da liễu, bệnh viện da liễu', '皮肤科', '皮膚科')\n"
            "- 한방과('Oriental Medicine', 'Дорно Дахины Анагаах Ухаан', 'đông y', '东方医学', '東方醫學')\n"
            "When provided with user symptoms, identify the most relevant department and translate it accurately. "
            "Respond in JSON format with the following keys:\n"
            "1) 'department': The most relevant medical department (translated into the user's language(as received in the 'Language' parameter) and Korean).\n"
            "2) 'possible_conditions': A list of one or more possible diseases (translated into the user's language(as received in the 'Language' parameter) and Korean). Use precise and formal medical terminology for all translations. Only include conditions that are meaningfully relevant, avoiding unnecessary or redundant entries.When translating, ensure that terms like '염증' are not simplified or shortened (e.g., do not translate '염증' to '염'). Additionally, avoid translating a condition as '암' (cancer) unless the condition clearly and accurately corresponds to cancer in all contexts and languages. Similarly, ensure that conditions such as '결막염' (conjunctivitis) or other non-cancerous diseases are not mistranslated as cancer-related terms (e.g., Нүдний хавдар) or other incorrect meanings in any language.\n"
            "3) 'questions_to_doctor': A list of questions the user should ask their doctor, considering the 'possible_conditions'. Provide up to five unique and meaningful questions, avoiding repetitive or semantically identical phrasing. Each question must be provided in both the user's language(as received in the 'Language' parameter) and Korean, focusing on the condition, symptoms, or treatment. Ensure that the questions are practical and directly related to the possible conditions.\n"
            "4) 'symptom_checklist': A checklist of detailed symptoms associated with each possible condition listed in 'possible_conditions'. Each entry in the checklist should correspond to a condition from 'possible_conditions' and include a list of up to five related symptom names translated into both Korean and the user's language(as received in the 'Language' parameter). Use precise and formal medical terms for symptom descriptions and avoid repetitive descriptions. Focus on symptoms that highlight the uniqueness or severity of the condition.\n"
            "Respond in both the user's language(as received in the 'Language' parameter) and Korean above all. Ensure all translations use accurate and formal medical terms rather than colloquial expressions. Minimize repetitive sentences, and focus on providing fewer but more meaningful and accurate responses."
        )


        #GPT API 호출
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User symptoms: {combined_description}. Language: {language}"}
            ],
            temperature=0.4
        )

        #응답 파싱
        result = response.choices[0].message.content
        return json.loads(result)

    except Exception as e:
        #예외 처리
        return {"error": str(e)}

def romanize_korean_names(names: list[str]) -> dict:
    """
    여러 병원명 또는 약국명을 GPT를 사용해 음독(로마자 표기)으로 한 번에 변환
    Args:
        names (list[str]): 한국어 병원명 리스트
    Returns:
        dict[str, str]: {병원명: 음독 결과}
    """
    import openai
    import json

    try:
        system_prompt = (
            "You are a Korean language expert. Convert the following Korean medical facility names into Romanized Korean "
            "using proper spacing. Always separate the medical suffix at the end like '병원', '의원', '약국', '한의원'.\n"
            "Return the result as a JSON dictionary where each key is the original name and the value is the Romanized name. "
            "No explanation, only valid JSON.\n"
            "Example:\n"
            "{\n"
            "  \"강현우비뇨기과의원\": \"Kanghyunu Binyogigwa Uiwon\",\n"
            "  \"해림온누리약국\": \"Haerim Onnuri Yakguk\"\n"
            "}"
        )

        name_list_text = "\n".join(f"- {name}" for name in names)

        user_prompt = f"Romanize the following names:\n{name_list_text}"

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"Romanization Error: {e}")
        return {}