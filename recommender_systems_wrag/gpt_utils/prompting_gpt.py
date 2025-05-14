from .department_mapping import get_department_translation
import openai
import json
import configparser
from rag_utils.rag_search import search_similar_conditions

config=configparser.ConfigParser()
config.read('keys.config')
openai.api_key=config['API_KEYS']['chatgpt_api_key']
def analyze_symptoms(symptoms, language):
    import openai, json, random
    
    #입력 검증
    if not symptoms or not isinstance(symptoms, list):
        raise ValueError("증상 정보는 비어있지 않은 리스트여야 합니다.")
    
    #RAG를 통한 유사 질병 검색
    similar_conditions=search_similar_conditions(symptoms)
    
    #증상 문자열 합치기
    symptom_description=""
    for s in symptoms:
        if not isinstance(s, dict):
            continue
            
        macro=", ".join(s.get('macro_body_parts', []) or [])
        micro=", ".join(s.get('micro_body_parts', []) or [])
        detail=s.get('symptom_details', {}) or {}
        
        if macro or micro or detail:
            symptom_description += f"macro: {macro}, micro: {micro}, details: {json.dumps(detail, ensure_ascii=False)} | "
    
    if not symptom_description.strip():
        raise ValueError("유효한 증상 정보가 없습니다.")

    base_questions={
        "KO": [
            "통증이 생기면 어떻게 해야 하나요?",
            "치료나 진료에 대해 궁금한 점이 있으면 누구에게 연락해야 하나요?",
            "제 증상의 원인이 무엇이라고 생각하시나요?",
            "이 문제를 진단하기 위해 어떤 검사를 하실 건가요?",
            "그 검사들은 얼마나 안전한가요?",
            "치료를 할 경우와 하지 않을 경우 장기적인 예후는 어떤가요?",
            "증상이 더 심해지면 제가 스스로 해야 할 일은 무엇이고, 언제 병원에 연락해야 하나요?",
            "피해야 할 음식이나 활동이 있을까요?",
            "입원이 필요한가요?"
        ],
        "EN": [
            "What do I do when I experience pain?",
            "Who do I contact when I have concerns about my care or services?",
            "What do you think is causing my problem?",
            "What tests will you do to diagnose the problem?",
            "How safe are the tests?",
            "What is the long-term outlook with and without treatment?",
            "If my symptoms get worse, what should I do on my own? When should I contact you?",
            "Are there any activities or foods I should avoid?",
            "Do I need to be hospitalized?"
        ],
        "VI": [
            "Tôi nên làm gì khi bị đau?",
            "Tôi nên liên hệ với ai khi có thắc mắc về việc điều trị hoặc dịch vụ?",
            "Bác sĩ nghĩ nguyên nhân gây ra vấn đề của tôi là gì?",
            "Bác sĩ sẽ làm những xét nghiệm nào để chẩn đoán vấn đề này?",
            "Các xét nghiệm đó có an toàn không?",
            "Tiên lượng lâu dài sẽ như thế nào nếu điều trị và nếu không điều trị?",
            "Nếu các triệu chứng trở nên nghiêm trọng hơn, tôi nên tự làm gì? Khi nào tôi nên liên hệ với bác sĩ?",
            "Có hoạt động hoặc thực phẩm nào tôi nên tránh không?",
            "Tôi có cần nhập viện không?"
        ],
        "ZH_CN": [
            "我感到疼痛时该怎么办？",
            "如果我对治疗或服务有疑问，应联系谁？",
            "您认为我的问题是什么原因导致的？",
            "您会做哪些检查来诊断这个问题？",
            "这些检查安全吗？",
            "治疗与不治疗的长期预后如何？",
            "如果症状加重，我该自己做些什么？什么时候应该联系您？",
            "有哪些活动或食物是我应该避免的？",
            "我需要住院吗？"
        ],
        "ZH_TW": [
            "我感到疼痛時該怎麼辦？",
            "如果我對治療或服務有疑問，應聯絡誰？",
            "您認為我的問題是什麼原因造成的？",
            "您會進行哪些檢查來診斷這個問題？",
            "這些檢查是否安全？",
            "接受與不接受治療的長期預後如何？",
            "如果症狀變嚴重，我應該自己做什麼？什麼時候應該聯絡您？",
            "我應該避免哪些活動或食物？",
            "我需要住院嗎？"
        ]
    }
    #질명별 질문 템플릿
    condition_question_templates={
        "KO": [
            "{}에 대한 치료는 일반적으로 얼마나 오래 걸리나요?",
            "{}에 의해 합병증이 생길 수도 있나요?",
            "{}(이)라면 병원에 다시 와야 하는 시점은 언제인가요?",
            "{}에 대해 보통 사람들이 잘못 이해하는 점이 있다면 알려주세요.",
            "{}일 경우 어떤 증상이 더 나타날 수 있나요?"
        ],
        "EN": [
            "How long does treatment for {} usually take?",
            "Can {} cause complications?",
            "When should I return to the hospital if I have {}?",
            "What do people often misunderstand about {}?",
            "What other symptoms may appear if I have {}?"
        ],
        "VI": [
            "Điều trị {} thường kéo dài bao lâu?",
            "{} có thể gây ra biến chứng không?",
            "Khi nào tôi nên quay lại bệnh viện nếu bị {}?",
            "Mọi người thường hiểu sai điều gì về {}?",
            "Nếu tôi bị {}, có thể xuất hiện những triệu chứng nào khác?"
        ],
        "ZH_CN": [
            "治疗{}通常需要多长时间？",
            "{}可能会引起并发症吗？",
            "如果我患有{}，什么时候需要回医院？",
            "人们通常对{}有哪些误解？",
            "如果我得了{}，还可能出现哪些其他症状？"
        ],
        "ZH_TW": [
            "治療{}通常需要多長時間？",
            "{}可能會引起併發症嗎？",
            "如果我患有{}，什麼時候應該回醫院？",
            "人們通常對{}有哪些誤解？",
            "如果我得了{}，還可能出現哪些其他症狀？"
        ]
    }
    
    prompt=(
        "You are a multilingual medical assistant."
        "Your task is to return the following fields in JSON format, with proper multilingual formatting:\n\n"
        "1) 'department_ko':Based ONLY on macro and micro body parts, return the most relevant Korean department (진료과) name." 
        "Ignore any other symptom details. Do NOT guess unrelated departments.Departments unrelated to macro body parts must NOT be selected."
        "Prioritize macro over micro body parts when determining the department. Choose only from the following Korean departments:\n"
        "   - 가정의학과\n"
        "   - 내과\n"
        "   - 마취통증의학과\n"
        "   - 비뇨의학과\n"
        "   - 산부인과\n"
        "   - 성형외과\n"
        "   - 소아청소년과\n"
        "   - 신경과\n"
        "   - 신경외과\n"
        "   - 심장혈관흉부외과\n"
        "   - 안과\n"
        "   - 영상의학과\n"
        "   - 예방의학과\n"
        "   - 외과\n"
        "   - 이비인후과\n"
        "   - 재활의학과\n"
        "   - 정신건강의학과\n"
        "   - 정형외과\n"
        "   - 치의과\n"
        "   - 피부과\n"
        "   - 한방과\n\n"

        "2) 'possible_conditions': A list of objects, each with a 'condition' field containing language-specific translations (e.g., {'KO': '무릎 관절염', 'VI': 'Viêm khớp gối'})."
        "Use the following similar conditions as reference:\n"
        f"{json.dumps(similar_conditions, ensure_ascii=False, indent=2)}\n\n"
        "3) 'questions_to_doctor': Leave this field as an empty list []. The questions will be filled in by the application logic later."
        "4) 'symptom_checklist': A list of objects. Each object must contain:\n"
        "   - 'condition_ko': the condition name in Korean\n"
        "   - 'condition_translation': a dict with keys 'KO' and user's language\n"
        "   - 'symptoms': a list of symptom translations, each as a dict with 'KO' and user's language\n\n"

        "Use only medically relevant conditions based on the department and symptoms. Use formal medical language.\n\n"

        f"Respond ONLY with valid JSON in the following structure:\n"
        "{\n"
        '  "department_ko": "정형외과", \n'
        '  "possible_conditions": [ {"condition": {"KO": "...", "' + language.upper() + '": "..."}} ],\n'
        '  "questions_to_doctor": [ {"KO": "...", "' + language.upper() + '": "..."} ],\n'
        '  "symptom_checklist": {\n'
        '    "무릎 관절염": {\n'
        '      "condition_translation": {"KO": "무릎 관절염", "' + language.upper() + '": "Viêm khớp gối"},\n'
        '      "symptoms": [ {"KO": "...", "' + language.upper() + '": "..."} ]\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "Respond ONLY with valid JSON. Do NOT include any explanation or formatting. No markdown.\n\n"
        "[LANGUAGE RULE]\n"
        "- If the user's language is \"KO\", return only Korean ('KO') in all translations. Do not include any other language keys.\n"
        "- Otherwise, always include both 'KO' and the user's language code (e.g., 'VI', 'EN', 'ZH') — and no more.\n"
        "- Never include keys for unused languages."
    )

    #실제 호출
    response=openai.chat.completions.create(
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

    #파싱
    result=json.loads(response.choices[0].message.content.strip())
    questions_to_doctor=[]

    base_count=min(3, len(base_questions[language]))
    condition_count=min(2, len(result["possible_conditions"]))

    #base 질문 처리
    for q in random.sample(range(len(base_questions[language])), base_count):
        if language == "KO":
            questions_to_doctor.append({"KO": base_questions["KO"][q]})
        else:
            questions_to_doctor.append({
                "KO": base_questions["KO"][q],
                language: base_questions[language][q]
            })

    #condition 질문 처리
    chosen_idxs=random.sample(range(len(result["possible_conditions"])), condition_count)
    template_idxs=random.sample(range(len(condition_question_templates[language])), 2)

    for i, cond_idx in enumerate(chosen_idxs):
        cond=result["possible_conditions"][cond_idx]["condition"]
        t_idx=template_idxs[i]

        if language == "KO":
            questions_to_doctor.append({
                "KO": condition_question_templates["KO"][t_idx].format(cond["KO"])
            })
        else:
            questions_to_doctor.append({
                "KO": condition_question_templates["KO"][t_idx].format(cond["KO"]),
                language: condition_question_templates[language][t_idx].format(cond[language])
            })

    result["questions_to_doctor"]=questions_to_doctor
    return result

def romanize_korean_names(names: list[str]) -> dict:
    #병원명이나 약국명을 GPT를 사용해 음독(로마자 표기)
    import openai
    import json

    try:
        system_prompt=(
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

        name_list_text="\n".join(f"- {name}" for name in names)

        user_prompt=f"Romanize the following names:\n{name_list_text}"

        response=openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        result_text=response.choices[0].message.content.strip()
        result=json.loads(result_text)
        return result

    except Exception as e:
        print(f"Romanization Error: {e}")
        return {}