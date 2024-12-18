import openai
import json
import configparser
from prompting_gpt import get_medical_info

config = configparser.ConfigParser()
config.read('C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

if __name__ == "__main__":
    # 테스트 증상 정보
    symptoms = {
        "macro_body_part": "Chest",
        "micro_body_part": "Left lung",
        "symptom_details": "Persistent cough, moderate pain, 3 days duration",
        "additional_info": "https://s3.amazonaws.com/bucket/image.jpg"
    }
    language = "vi"  # 사용자 언어: en(영어), mn(몽골어), zh(중국어-간체), 
    #zh-TW(중국어-번체), vi(베트남어)

    # GPT API 호출
    result = get_medical_info(symptoms, language)

    # 결과 출력
    print(json.dumps(result, indent=4, ensure_ascii=False))