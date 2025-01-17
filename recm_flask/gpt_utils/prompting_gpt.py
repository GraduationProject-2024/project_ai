import openai
import json
import os

openai.api_key = os.getenv("CHATGPT_API_KEY")

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
        # 증상 정보를 문자열로 변환
        symptom_descriptions = []
        for symptom in symptoms:
            symptom_descriptions.append(
                f"Macro body part: {symptom.get('macro_body_part', 'N/A')}, "
                f"Micro body part: {symptom.get('micro_body_part', 'N/A')}, "
                f"Details: {symptom.get('symptom_details', 'N/A')}, "
                f"Additional info: {symptom.get('additional_info', 'N/A')}"
            )

        # 모든 증상 정보를 하나의 설명으로 결합
        combined_description = " | ".join(symptom_descriptions)


        # 프롬프트 설정
        prompt = (
            "You are a multilingual medical assistant specializing in professional medical terminology. "
            "When translating symptoms into Korean, ensure the following translations are used:\n"
            "- 'swelling' should be translated as '붓기' (not '부기').\n"
            "Below is a list of valid medical departments with accurate names in Korean:\n"
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
            "- 치의과(it menas dental hospital or dental clinic.)\n"
            "- 피부과\n"
            "When provided with user symptoms, identify the most relevant department and translate it accurately. "
            "Respond in JSON format with the following keys:\n"
            "1) 'department': The most relevant medical department (translated into the user's language and Korean).\n"
            "2) 'possible_conditions': A list of possible conditions or diseases.\n"
            "3) 'questions_for_doctor': A list of questions the user should ask their doctor. It is consist of 5 questions. Each question must be provided in both the user's language and Korean.\n"
            "4) 'symptom_checklist': A detailed checklist of symptoms associated with each possible condition listed in 'possible_conditions'. Each entry in the checklist should correspond to a condition from 'possible_conditions' and include a list of up to five symptom names translated into both Korean and the user's language. Symptoms should reflect the severity and uniqueness of the condition, while common symptoms can be shared among conditions.\n"
            "Respond in both the user's language and Korean above all. Ensure all translations use the valid terms provided in the list above."
            "But, if the user's language is 'ko', respond only in Korean once without trying to translate into other languages."
            )

        # GPT API 호출
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User symptoms: {combined_description}. Language: {language}"}
            ],
            temperature=0.5
        )

        # 응답 파싱
        result = response.choices[0].message.content
        return json.loads(result)

    except Exception as e:
        # 예외 처리
        return {"error": str(e)}