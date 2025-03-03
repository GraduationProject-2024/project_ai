import openai
import configparser
# API 키 설정
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']



def generate_title_and_type(content):
    """
    신고 내용을 기반으로 적절한 제목(한글 & 영어)과 emergency_type을 생성
    """
    prompt = f"""
    Analyze the following emergency report and generate:
    1. A concise and descriptive title in Korean.
    2. The same title translated into English.
    3. The appropriate emergency type from the following categories: Fire, Salvage, Emergency, Traffic Accident, Disaster. 
       If the content does not fit any of these categories, return 'Etc'.

    Return the results in JSON format:
    {{
        "title_ko": "Korean title",
        "title_en": "English title",
        "emergency_type": "Selected category"
    }}

    ---
    {content}
    ---
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=150
        )
        result = response.choices[0].message.content.strip()

        # ✅ JSON 변환 (예외 처리)
        import json
        try:
            data = json.loads(result)
            title_ko = data.get("title_ko", "긴급 신고")
            title_en = data.get("title_en", "Emergency Report")
            emergency_type = data.get("emergency_type", "Etc")  # 기본값 "Etc"
        except json.JSONDecodeError:
            print("❌ JSON 변환 실패, 기본값 반환")
            title_ko = "긴급 신고"
            title_en = "Emergency Report"
            emergency_type = "Etc"

        return title_ko, title_en, emergency_type

    except Exception as e:
        print(f"❌ 제목 및 유형 생성 실패: {e}")
        return "긴급 신고", "Emergency Report", "Etc"

import json
def summarize_content(content):
    """
    신고 내용을 영어와 한국어로 번역 & 요약 (총 800 bytes 이내)
    - 모든 언어를 감지하여 영어와 한국어로 변환
    - 욕설 및 불쾌한 표현을 필터링하여 정제
    """
    prompt = f"""
    Analyze the following report and generate a refined summary in both Korean and English.
    - Ensure that any vulgar, insulting, or offensive language is removed while preserving the original meaning.
    - Translate the content into natural Korean and English.
    - The combined length of both summaries should not exceed 800 bytes.
    - Adjust the length of each language appropriately to fit within the limit.
    
    Return the results in JSON format:
    {{
        "summary_ko": "Summarized content in Korean",
        "summary_en": "Summarized content in English"
    }}
    
    ---
    {content}
    ---
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=600  # 🚀 최대 800 bytes 제한을 고려하여 적절한 토큰 설정
        )
        result = response.choices[0].message.content.strip()

        data = json.loads(result)
        summary_ko = data.get("summary_ko", "요약된 신고 내용이 없습니다.")
        summary_en = data.get("summary_en", "No summarized report available.")

        return summary_ko, summary_en

    except Exception as e:
        print(f"❌ 요약 실패: {e}")
        return "요약된 신고 내용이 없습니다.", "No summarized report available."