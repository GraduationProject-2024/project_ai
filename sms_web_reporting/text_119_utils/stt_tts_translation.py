import openai
from .s3_utils import upload_to_s3

import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']


def transcribe_audio(audio_file_path):

    from .s3_utils import download_from_s3
    """
    Whisper를 사용하여 음성을 텍스트로 변환 후 번역
    :param audio_file_path: 업로드된 오디오 파일 경로
    :param target_language: 번역할 대상 언어 (기본값: "ko" - 한국어)
    :return: 변환된 텍스트
    """
    if audio_file_path.startswith("http"):
         print(f"S3 URL 감지: {audio_file_path}")
         audio_file_path = download_from_s3(audio_file_path)
         print(f"다운로드된 로컬 파일 경로: {audio_file_path}")

    with open(audio_file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                #model="whisper-1",
                model="gpt-4o-transcribe",
                file=audio_file
            )
    
    return response.text, audio_file_path

def translate_and_filter_text(text, target_language="ko"):
    """
    GPT API를 사용하여 문장을 원하는 언어로 번역
    :param text: 입력된 텍스트
    :param target_language: 번역할 대상 언어 (예: "ko", "en", "vi", "zh")
    :return: 번역된 텍스트
    """
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


from datetime import datetime
import tempfile

def generate_tts(text):
    """
    사용자의 언어에 맞게 TTS 음성을 생성하는 함수
    :param text: 변환할 텍스트
    :param use_openai: OpenAI TTS 사용 여부 (기본값: True)
    :return: 생성된 음성 파일 경로
    """
    #파일 이름을 미리 생성 (S3 업로드에 필요)
    timestamp = int(datetime.now().timestamp())
    file_name = f"{timestamp}.mp3"
    
    #임시 파일 생성
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio_path = temp_audio.name  #파일 경로 저장
    temp_audio.close()  #파일 닫기 (스트림 해제)

    #OpenAI TTS 호출
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(temp_audio_path)  #TTS 음성 파일 저장

        #S3 업로드 시 직접 파일 이름을 지정하여 전달
        with open(temp_audio_path, "rb") as audio_file:
            audio_file.filename = file_name  #filename 속성 추가 (upload_to_s3과 호환)
            s3_url = upload_to_s3(audio_file, "audio/tts_outputs/")

        #임시 파일 삭제
        import os
        os.remove(temp_audio_path)

        return s3_url
    except Exception as e:
        return {"status": "error", "message": str(e)}