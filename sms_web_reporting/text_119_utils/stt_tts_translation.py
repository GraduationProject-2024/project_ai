import openai
from .s3_utils import upload_to_s3

import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']


def transcribe_audio(audio_file_path):
    #음성을 텍스트로 변환 후 번역
    from .s3_utils import download_from_s3
    if audio_file_path.startswith("http"):
         print(f"S3 URL: {audio_file_path}")
         audio_file_path = download_from_s3(audio_file_path)
         print(f"다운로드된 로컬 파일 경로: {audio_file_path}")

    with open(audio_file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
    
    return response.text, audio_file_path

def translate_and_filter_text(text, target_language="ko"):
    #GPT API로 번역 및 내용 정제
    language_map = {
        "ko":"Korean",
        "en":"English",
        "vi":"Vietnamese",
        "zh":"Chinese(Simplified)",
        "zh-hant":"Chinese(Traditional)"
    }
    language = language_map.get(target_language)

    #번역 프롬프트 설정
    prompt = (
         f"Check if the following text is already translated into {language}. "
        "If it is not translated, translate it to {language}. "
        "If it is already in {language}, do not translate it but refine any vulgar, insulting, or otherwise offensive language. "
         "If the text contains vulgar, insulting, or otherwise offensive language, please refine these expressions into a more polite and respectful form while preserving the original context and intended meaning. "
         "Do not explain or provide additional interpretations. "
         "Maintain the original meaning without adding explanations about informal expressions, slang, or grammatical irregularities. "
         "Ensure that idioms and colloquial language are translated naturally and concisely. "
         "Use commonly understood loanwords where appropriate. "
         "If the text contains cultural expressions, translate them in a way that conveys the intended meaning accurately, but do not over-explain them."
    )

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
            ]
    ) 
    
    return response.choices[0].message.content