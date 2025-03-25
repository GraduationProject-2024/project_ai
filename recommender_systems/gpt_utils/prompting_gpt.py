from .department_mapping import get_department_translation
import openai
import json
import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

def get_department(symptoms, language):
    combined_description = ""
    for symptom in symptoms:
        macro = ", ".join(symptom.get('macro_body_parts', []))
        micro = ", ".join(symptom.get('micro_body_parts', []))
        combined_description += f"Macro: {macro}, Micro: {micro} | "

    prompt = (
        "You are a multilingual medical assistant.\n"
        "Based ONLY on macro and micro body parts, return the most relevant Korean department (진료과) name.\n"
        "Ignore any other symptom details. Do NOT guess unrelated departments.\n"
        "Departments unrelated to macro body parts must NOT be selected. Prioritize macro over micro body parts when determining the department.\n"
        "Choose only from the following Korean departments:\n"
        "- 가정의학과\n"
        "- 내과\n"
        "- 마취통증의학과\n"
        "- 비뇨의학과\n"
        "- 산부인과\n"
        "- 성형외과\n"
        "- 소아청소년과\n"
        "- 신경과\n"
        "- 신경외과\n"
        "- 심장혈관흉부외과\n"
        "- 안과\n"
        "- 영상의학과\n"
        "- 예방의학과\n"
        "- 외과\n"
        "- 이비인후과\n"
        "- 재활의학과\n"
        "- 정신건강의학과\n"
        "- 정형외과\n"
        "- 치의과\n"
        "- 피부과\n"
        "- 한방과\n\n"
        "Return JSON only: { \"department_ko\": \"정형외과\" }"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Symptoms: {combined_description}\nLanguage: {language}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.3
    )

    result = json.loads(response.choices[0].message.content.strip())
    dept_ko = result["department_ko"]

    return get_department_translation(dept_ko, language)

# def get_department(symptoms, language):
#     import openai, json
    
#     combined_description = ""
#     for symptom in symptoms:
#         macro = ", ".join(symptom.get('macro_body_parts', []))
#         micro = ", ".join(symptom.get('micro_body_parts', []))
#         combined_description += f"Macro: {macro}, Micro: {micro} | "

#     prompt = (
#         "You are a multilingual medical assistant. You will receive a description of user symptoms, including:\n"
#         "- macro_body_parts (e.g., 무릎, 다리)\n"
#         "- micro_body_parts (e.g., 정강이, 발목)\n"
#         "- symptom_details (used ONLY to assess urgency, NOT to guess unrelated conditions)\n\n"

#         "Your job is to identify the most medically relevant department (진료과) **only** from the list below. "
#         "The department must be directly related to the macro/micro body parts mentioned. "
#         "**Do not use symptom_details to infer unrelated conditions**. "
#         "**Do not suggest any department that is unrelated to the body parts.** For example, if the symptoms mention only '무릎' or '발', you should NOT return '내과'.\n\n"

#         "Below is a list of valid medical departments with accurate names in Korean(English, Vietnamese, Chinese, Chinese-Hant):"
#         "- 가정의학과('Family Medicine', 'y học gia đình', '家庭医学科', '家庭醫學')\n"
#         "- 내과('Internal Medicine', 'khoa nội, bệnh viện nội khoa', '内科', '內科')\n"
#         "- 마취통증의학과('Anaesthesiology', 'khoa chứng đau gây mê', '麻醉疼痛医学科', '麻醉痛医学科')\n"
#         "- 비뇨의학과('Urology', 'khoa tiết niệu', '泌尿医学系', '泌尿学系')\n"
#         "- 산부인과('Obstetrics and Gynecology', 'khoa phụ sản, bệnh viện phụ sản', '妇产科', '婦產科')\n"
#         "- 성형외과('Plastic&Reconstructive Surgery', 'Phẫu thuật tạo hình và tái tạo', '整形及重建外科', '整形及重建外科')\n"
#         "- 소아청소년과('Pediatrics', 'khoa nhi', '儿童青少年科', '小儿青少年科')\n"
#         "- 신경과('Neurology', 'Thần kinh học', '神经科', '神经科')\n"
#         "- 신경외과('Neurological Surgery', 'khoa ngoại thần kinh, bệnh viện ngoại khoa', '神经外科', '神经外科')\n"
#         "- 심장혈관흉부외과('Thoracic Surgery', 'khoa ngoại khoa tim mạch', '心血管胸外科', '心血管胸外科')\n"
#         "- 안과('Ophthalmology', 'nhãn khoa, bệnh viện mắt', '眼科', '眼科')\n"
#         "- 영상의학과('Imaging Radiology', 'ngành X-quang', '影像医学科', '影像放射學')\n"
#         "- 예방의학과('Preventive Medicine', 'Y học dự phòng', '预防医学科', '預防醫學')\n"
#         "- 외과('General Surgery', 'khoa ngoại, bệnh viện ngoại khoa', '外科', '一般外科')\n"
#         "- 이비인후과('Otolaryngology', 'khoa tai mũi họng, bệnh viện tai mũi họng', '耳鼻喉科', '耳鼻喉科')\n"
#         "- 재활의학과('Rehabilitation Medicine', 'thuốc phục hồi chức năng', '康复医学系', '康复医法系')\n"
#         "- 정신건강의학과('Psychiatry', 'Tâm thần học', '心理健康医学系', '精神健康医学系')\n"
#         "- 정형외과('Orthopedic Surgery', 'khoa ngoại chỉnh hình, bệnh viện chấn thương chỉnh hình', '骨科手术', '骨科手術')\n"
#         "- 치의과('Dentistry', 'nha khoa, bệnh viện nha khoa', '牙科', '牙科')\n"
#         "- 피부과('Dermatology', 'khoa da liễu, bệnh viện da liễu', '皮肤科', '皮膚科')\n"
#         "- 한방과('Oriental Medicine', 'đông y', '东方医学', '東方醫學')\n"
        
#         "Return ONLY ONE department, as a JSON object with language-specific keys.\n"
#         "Example: { \"department\": { \"KO\": \"정형외과\", \"VI\": \"khoa ngoại chỉnh hình\" } }\n\n"
#         "Example: { \"department\": { \"KO\": \"이비인후과\", \"EN\": \"Otolaryngology\" } }\n\n"
#         "- If the user's language is \"KO\", return only the 'KO' key.\n"
#         "- Otherwise, return both 'KO' and the user's language key (e.g., 'EN', 'VI', 'ZH').\n\n"
#         "Respond ONLY with valid JSON. No explanation. No formatting."
#         )

#     messages = [
#         {"role": "system", "content": prompt},
#         {
#             "role": "user", 
#             "content": f"Symptoms: {combined_description}\nLanguage: {language}"
#         }
#     ]

#     response = openai.chat.completions.create(
#         model="gpt-4",
#         messages=messages,
#         temperature=0.3
#     )
#     result = response.choices[0].message.content
#     return json.loads(result)["department"]


def get_condition_details(symptoms, language, department):
    import openai, json

    """
    department (str): 이미 get_department에서 받은 국문 진료과. ex) "정형외과"
    returns: {
       "possible_conditions": [...],
       "questions_to_doctor": [...],
       "symptom_checklist": {...}
    }
    """

    # 증상 문자열 합치기(간단 예)
    symptom_description = ""
    for s in symptoms:
        macro = ", ".join(s.get('macro_body_parts', []))
        micro = ", ".join(s.get('micro_body_parts', []))
        detail = s.get('symptom_details', {})
        symptom_description += f"macro: {macro}, micro: {micro}, details: {detail} | "

    # 프롬프트: department는 이미 정해졌으니 이걸 참고하라고 안내
    # prompt = (
    #     "You are a multilingual medical assistant. The user symptoms and department have already been established.\n"
    #     f"Department: {department}\n\n"

    #     "Your task is to return the following fields in JSON format, with proper multilingual formatting:\n\n"

    #     "1) 'possible_conditions': A list of objects, each with a 'condition' field containing language-specific translations (e.g., {'KO': '무릎 관절염', 'VI': 'Viêm khớp gối'})\n"
    #     "2) 'questions_to_doctor': A list of up to five practical and specific questions that the user (as a patient) should ask a doctor during consultation. Each question must be an object with keys 'KO' and the user's language (e.g., 'VI').\n"
    #     "   - Each question must reflect the user's point of view (not the doctor's) and should help them understand the condition, treatment, risks, or follow-up steps. "
    #     "   - Questions should begin with phrases like “Do I need...”, “What should I...”, “Is it normal that...”, “Should I avoid...”, etc. "
    #     "   - Do NOT include questions that sound like something the doctor would say or explain unprompted. These are patient questions only.\n"
    #     "3) 'symptom_checklist': For each condition (use Korean name as the key), provide:\n"
    #     "   - 'symptoms': a list of symptoms with translations, each as a dict like {'KO': '무릎 통증', 'VI': 'Đau đầu gối'}\n"
    #     "   - 'condition_translation': a dict with 'KO' and the user's language, representing the condition name translation.\n\n"
        
    #     "Use formal medical terminology only. Avoid guessing unrelated conditions. Use symptom_details only to judge severity.\n\n"

    #     f"Respond ONLY with valid JSON in the following structure:\n"
    #     "{\n"
    #     '  "possible_conditions": [ {"condition": {"KO": "...", "' + language.upper() + '": "..."}} ],\n'
    #     '  "questions_to_doctor": [ {"KO": "...", "' + language.upper() + '": "..."} ],\n'
    #     '  "symptom_checklist": {\n'
    #     '    "무릎 관절염": {\n'
    #     '      "condition_translation": {"KO": "무릎 관절염", "' + language.upper() + '": "Viêm khớp gối"},\n'
    #     '      "symptoms": [ {"KO": "...", "' + language.upper() + '": "..."} ]\n'
    #     "    }\n"
    #     "  }\n"
    #     "}\n\n"
    #     "Respond ONLY with valid JSON. Do NOT include any explanation or formatting. No markdown.\n\n"
    #     "[LANGUAGE RULE]\n"
    #     "- If the user's language is \"KO\", return only Korean ('KO') in all translations. Do not include any other language keys.\n"
    #     "- Otherwise, always include both 'KO' and the user's language code (e.g., 'VI', 'EN', 'ZH') — and no more.\n"
    #     "- Never include keys for unused languages."
    # )
    prompt = (
        "You are a multilingual medical assistant. The user symptoms and department have already been established.\n"
        f"Department: {department}\n\n"

        "Your task is to return the following fields in JSON format:\n\n"

        "1) 'possible_conditions': A list of objects, each with a 'condition' field containing language-specific translations (e.g., {'KO': '무릎 관절염', 'VI': 'Viêm khớp gối'})\n"
        "2) 'questions_to_doctor': A list of up to five patient-centered questions to ask the doctor. Each question must be an object with keys 'KO' and the user's language (e.g., 'VI').\n"
        "3) 'symptom_checklist': A list of objects. Each object must contain:\n"
        "   - 'condition_ko': the condition name in Korean\n"
        "   - 'condition_translation': a dict with keys 'KO' and user's language\n"
        "   - 'symptoms': a list of symptom translations, each as a dict with 'KO' and user's language\n\n"

        "Use only medically relevant conditions based on the department and symptoms. Use formal medical language.\n\n"

        f"Return valid JSON in this format:\n"
        "{\n"
        '  "possible_conditions": [ {"condition": {"KO": "...", "' + language.upper() + '": "..."}} ],\n'
        '  "questions_to_doctor": [ {"KO": "...", "' + language.upper() + '": "..."} ],\n'
        '  "symptom_checklist": [\n'
        "    {\n"
        '      "condition_ko": "무릎 관절염",\n'
        '      "condition_translation": {"KO": "무릎 관절염", "' + language.upper() + '": "Viêm khớp gối"},\n'
        '      "symptoms": [ {"KO": "무릎 통증", "' + language.upper() + '": "Đau đầu gối"} ]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "[LANGUAGE RULE]\n"
        "- If language is 'KO', return only Korean keys.\n"
        "- Otherwise, include both 'KO' and the user's language key.\n"
        "- Never include extra languages.\n"
        "Respond with only valid JSON. No formatting. No explanations."
    )
    # 실제 호출
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Symptoms: {symptom_description}\nLanguage: {language}"
            }
        ],
        temperature=0.3
    )

    # 파싱
    result = response.choices[0].message.content.strip()
    return json.loads(result)

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