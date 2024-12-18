import openai
import json
import configparser
# ConfigParser 초기화
config = configparser.ConfigParser()
# keys.config 파일 읽기
config.read('C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config')
# OpenAI API 키 설정
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
        language (str): 사용자 언어 코드 (e.g., 'en', 'ko')
    
    Returns:
        dict: 진료과, 의심되는 질병, 의사에게 할 질문 리스트가 포함된 JSON 객체
    """
    try:
        # 증상 정보를 문자열로 변환
        symptom_description = (
            f"Macro body part: {symptoms.get('macro_body_part', 'N/A')}, "
            f"Micro body part: {symptoms.get('micro_body_part', 'N/A')}, "
            f"Details: {symptoms.get('symptom_details', 'N/A')}, "
            f"Additional info: {symptoms.get('additional_info', 'N/A')}"
        )

        # 프롬프트 설정
        prompt = (
            "You are a multilingual medical assistant specializing in professional medical terminology. "
            "Below is a list of valid medical departments with accurate names in Korean:\n"
            "- 가정의학과\n"
            "- 결핵과\n"
            "- 구강내과\n"
            "- 구강병리과\n"
            "- 구강악안면외과\n"
            "- 내과\n"
            "- 마취통증의학과\n"
            "- 방사선종양학과\n"
            "- 병리과\n"
            "- 비뇨의학과\n"
            "- 사상체질과\n"
            "- 산부인과\n"
            "- 성형외과\n"
            "- 소아청소년과\n"
            "- 소아치과\n"
            "- 신경과\n"
            "- 신경외과\n"
            "- 심장혈관흉부외과\n"
            "- 안과\n"
            "- 영상의학과\n"
            "- 영상치의학과\n"
            "- 예방의학과\n"
            "- 예방치과\n"
            "- 외과\n"
            "- 응급의학과\n"
            "- 이비인후과\n"
            "- 재활의학과\n"
            "- 정신건강의학과\n"
            "- 정형외과\n"
            "- 직업환경의학과\n"
            "- 진단검사의학과\n"
            "- 치과\n"
            "- 치과교정과\n"
            "- 치과보존과\n"
            "- 치과보철과\n"
            "- 치주과\n"
            "- 침구과\n"
            "- 통합치의학과\n"
            "- 피부과\n"
            "- 한방내과\n"
            "- 한방부인과\n"
            "- 한방소아과\n"
            "- 한방신경정신과\n"
            "- 한방안·이비인후·피부과\n"
            "- 한방응급\n"
            "- 한방재활의학과\n"
            "- 핵의학과\n"
            "When provided with user symptoms, identify the most relevant department and translate it accurately. "
            "Respond in JSON format with the following keys:\n"
            "1) 'department': The most relevant medical department (translated into the user's language and Korean).\n"
            "2) 'possible_conditions': A list of possible conditions or diseases.\n"
            "3) 'questions_for_doctor': A list of questions the user should ask their doctor.\n"
            "Respond in both the user's language and Korean. Ensure all translations use the valid terms provided in the list above."
        )

        # GPT API 호출
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User symptoms: {symptom_description}. Language: {language}"}
            ],
            temperature=0.5
        )

        # 응답 파싱
        result = response.choices[0].message.content
        return json.loads(result)

    except Exception as e:
        # 예외 처리
        return {"error": str(e)}