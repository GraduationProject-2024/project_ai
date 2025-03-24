import openai
import configparser
import json
import boto3
import tempfile
import openai
import os
import pytz
from datetime import datetime

config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

import base64
def encode_audio_to_base64(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")
    
def transcribe_audio(file_path):
    """
    OpenAI Whisper를 사용하여 음성을 텍스트로 변환
    :param file_path: 변환할 오디오 파일 경로
    :return: 변환된 텍스트
    """
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt="이 오디오는 환자가 얘기하거나 의사가 얘기하는 내용입니다. 이를 고려해서 transcribe 해주세요."
        )
    return transcript.text

LANGUAGE_MAP = {
    #"af": "Afrikaans",
    #"ar": "Arabic",
    #"bg": "Bulgarian",
    #"bn": "Bengali",
    #"ca": "Catalan",
    #"cs": "Czech",
    #"cy": "Welsh",
    #"da": "Danish",
    #"de": "German",
    #"el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    #"hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese(Simplified)",
    "zh-tw": "Chinese(Traditional)"
}

from langdetect import detect
def detect_language(text, gpt_detected=None):
    """
    입력된 텍스트의 언어를 감지하여 언어명 반환
    GPT 감지 결과가 주어지면 이를 우선 적용
    """
    try:
        #GPT 감지 결과가 있으면 우선 사용
        if gpt_detected:
            print(f"GPT 감지 결과 사용: {gpt_detected}")
            return gpt_detected if gpt_detected in LANGUAGE_MAP.values() else None

        #langdetect를 통한 감지
        lang_code = detect(text)
        print(f"Detected language code: {lang_code}")  #감지된 코드 확인
        
        result = LANGUAGE_MAP.get(lang_code)
        if result is None:
            print(f"매핑되지 않은 언어 코드: {lang_code}")  #디버깅용 로그 추가
        return result
    except Exception as e:
        print(f"언어 감지 실패: {e}")
        return None 

def summarize_text(text):
    prompt = f"""
        Summarize the following conversation based on the **Original text** from the transcripts.  
        Ensure that the summary captures key points, emotions, and main ideas.  
        Provide the summary in **both Korean and English**.
        
        Some sentences may begin with a tag such as "Doctor:" or "Patient:". 
        These tags indicate who might be speaking, but they are not always accurate. 
        Use them only as helpful context, not as definitive labels.
        
        Original Text:
        {text}

        Respond strictly in the following JSON format **without additional text or explanations**:
        {{
        "summary_korean": "<summary_in_korean>",
        "summary_english": "<summary_in_english>"
        }}
        """

    response = openai.chat.completions.create(
        model="gpt-4", messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


def translate_text(text, previous_languages=[]):
    """
    입력된 텍스트에서 사용된 언어를 감지하고, 기존 감지된 언어를 고려하여 번역을 수행.
    :param text: 번역할 원본 텍스트
    :param previous_languages: 세션에서 이전에 감지된 언어 리스트
    :return: 감지된 언어 리스트, 번역 결과 (한국어, 영어, 추가 감지 언어 포함)
    """
    
    prompt = f"""
    Detect the language of the following text and return a JSON response **without any formatting such as markdown or code blocks**. 
    Then, translate it into Korean and English.  
    Also, include the original text under the key of the detected language (e.g., "Chinese") even if it's not Korean or English.  
    If there are any previously detected languages, also translate into those and include them in the 'translations' object.

    Previously detected languages: {', '.join(previous_languages) if previous_languages else 'None'}.

    Text: {text}

    Respond strictly in the following JSON format **without additional text or explanations**:
    {{
    "detected_language": "<detected_language>",
    "translations": {{
        "Korean": "<translated_text>",
        "English": "<translated_text>"
    }}
    }}

    If the detected language is different from Korean or English, add it to the 'translations' object using its language name as the key. 
    Additionally, if there are any previously detected languages, also add them to the 'translations' object. 
    For example, if the detected language is 'Vietnamese' and a previously detected language was 'Chinese', return:
    {{
    "detected_language": "Vietnamese",
    "translations": {{
        "Korean": "<translated_text>",
        "English": "<translated_text>",
        "Vietnamese": "<original_text>",
        "Chinese": "<translated_text>"
    }}
    }}
    Do NOT skip the detected language in the 'translations' object. Always include it.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.choices[0].message.content.strip()
    print(f"GPT Response:\n{response_text}")  #디버깅용

    try:
        result = json.loads(response_text)
        source_language = result.get("detected_language")

        #'source_language'도 번역 대상 언어에 추가
        detected_languages = list(set(previous_languages + [source_language]))
        translations = result.get("translations", {})

        #None 값을 빈 JSON '{}'으로 변환하여 MongoDB 저장 시 일관성 유지
        for lang, translation in translations.items():
            if translation is None or not translation.strip():
                translations[lang] = {}

    except json.JSONDecodeError:
        print("GPT 응답을 JSON으로 변환할 수 없음.")
        #한 번만 재시도
        response_retry = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            result = json.loads(response_retry.choices[0].message.content.strip())
        except json.JSONDecodeError:
            print("재시도 실패. fallback 적용.")
            return {
                "source_language": detect_language(text),
                "detected_languages": previous_languages,
                "translations": {}
            }

    return {
        "source_language": source_language,
        "detected_languages": detected_languages,  #'source_language'를 포함한 언어 목록 유지
        "translations": translations
    }



S3_BUCKET_NAME = config['S3_INFO']['BUCKET_NAME']


s3_client = boto3.client("s3",
                        aws_access_key_id=config['S3_INFO']['ACCESS_KEY_ID'],
                        aws_secret_access_key=config['S3_INFO']['SECRET_ACCESS_KEY'],
                        region_name="ap-northeast-2")
#AWS S3 설정
def upload_to_s3(file, folder, file_name):
    """
    파일을 AWS S3에 업로드
    :param file: 업로드할 파일 객체
    :param folder: S3 내 폴더 경로 (예: "audio/tts_outputs/recording/{lang}/")
    :param file_name: 저장할 파일 이름
    :return: S3 URL
    """
    s3_file_path = f"{folder}{file_name}"
    s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_file_path)
    
    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_file_path}"
    print(f"파일 업로드 완료: {s3_url}")
    return s3_url


def generate_tts(text, lang='ko'):
    """
    입력된 텍스트를 OpenAI TTS로 변환하고, S3에 업로드한 후 URL을 반환하는 함수.
    :param text: 변환할 텍스트
    :param lang: 음성 생성 언어 (기본값: "ko")
    :return: S3 URL 또는 오류 메시지
    """
    try:
        #1.KST(한국 시간) 기준으로 timestamp 생성
        kst = pytz.timezone("Asia/Seoul")
        timestamp = int(datetime.now(kst).timestamp())
        file_name = f"{timestamp}.mp3"

        #2️.임시 파일 생성
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()  #파일 닫기 (스트림 해제)

        #3️.OpenAI TTS 호출 및 저장
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(temp_audio_path)

        #4️.S3 업로드
        with open(temp_audio_path, "rb") as audio_file:
            audio_file.filename = file_name  #S3 업로드 시 필요한 파일 이름 속성 추가
            s3_url = upload_to_s3(audio_file, f"audio/tts_outputs/recording/{lang}/", file_name)

        #5️.임시 파일 삭제
        os.remove(temp_audio_path)

        return s3_url
    except Exception as e:
        return {"status": "error", "message": str(e)}



def classify_speaker(transcript):
    """
    주어진 텍스트가 의사 또는 환자의 발화인지 GPT로 판단하여 태그를 반환.
    :param transcript: 변환된 텍스트
    :return: 'Doctor', 'Patient', or 'Unknown'
    """
    try:
        prompt = f"""
        This audio is a spoken sentence from either a doctor or a patient.
        Determine who is speaking in the following sentence: "{transcript}"
        If it's the doctor, reply with "Doctor".
        If it's the patient, reply with "Patient".
        Only respond with one of these two values.
        """
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        tag = response.choices[0].message.content.strip()
        return tag if tag in ["Doctor", "Patient"] else "Unknown"
    except Exception as e:
        print(f"[ERROR] classify_speaker 실패: {e}", flush=True)
        return "Unknown"