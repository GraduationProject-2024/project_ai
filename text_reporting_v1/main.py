from flask import request, jsonify, Flask
from text_119_utils.stt_tts_translation import transcribe_audio, translate_and_filter_text, generate_tts
from text_119_utils.sms import send_messages

from text_119_utils.s3_utils import upload_to_s3
import os
#from text_119_utils.detect_language import detect_language

app = Flask(__name__)

#119 text reporting
@app.route('/send_messages', methods=['POST'])
def send_messages_route():
    """
    SMS 신고 전송 API (외국어 입력도 지원, 이미지 첨부 가능)
    - payload 예시:
        {
            "phone_number": "+8201051321887",
            "text": "Hello, I need help.",
            "audio": (optional)
        }
    """
    phone_number = request.form.get("phone_number")
    text = request.form.get("text", '')
    audio = request.files.get("audio")

    if not phone_number or not text:
        return jsonify({"status": "error", "message": "전화번호와 메시지는 필수입니다."}), 400

    local_audio_path = None  # 기본값 설정

    if audio:
        audio_path=upload_to_s3(audio, 'audio/textreporting/')
        text, local_audio_path = transcribe_audio(audio_path)
        
    
    # 번역 및 욕설 필터링
    #detected_lang = detect_language(text)

    processed_text = translate_and_filter_text(text) #if detected_lang != "ko" else text
    header_text = "[이 문자는 외국인 사용자 위주의 의료 서비스 앱, Mediko에서 번역 및 필터링 후 국제 SMS로 송신되었습니다.]\n"
    processed_text = header_text + processed_text

    
    result = send_messages(phone_number, processed_text)

    # SMS 전송 후 로컬 파일 삭제
    if local_audio_path is not None:
        try:
            os.remove(local_audio_path)
            print(f"로컬 오디오 파일 삭제 완료: {local_audio_path}")
        except Exception as e:
            print(f"로컬 오디오 파일 삭제 실패: {e}")

    return jsonify(result)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    업로드된 음성 파일을 STT로 변환 후 번역 (옵션)
    - 요청 형식:
        {
            "audio": (파일),
            "target_language": "ko" (선택 사항, 기본값: "ko")
        }
    - 응답:
        {
            "status": "success",
            "text": "번역된 텍스트"
        }
    """
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "오디오 파일이 필요합니다."}), 400


    audio_file = request.files['audio']

    s3_url = upload_to_s3(audio_file, "audio/transcript/")
    print(f"S3에 업로드된 파일 URL:{s3_url}")

    target_language = request.form.get("target_language", "ko")

    # **1) 원본 텍스트 추출 (STT)**
    original_text, local_audio_path = transcribe_audio(s3_url)  # 번역 없이 원본 텍스트 반환
    os.remove(local_audio_path)

    # **2) 번역된 텍스트 생성**
    translated_text = translate_and_filter_text(original_text, target_language)


    return jsonify({
        "status": "success",
        "original_text": original_text,
        "translated_text": translated_text,
        "audio_url": s3_url
    })


@app.route('/translate', methods=['POST'])
def translate():
    """
    텍스트 번역 API
    - 요청 형식:
        {
            "text": "번역할 텍스트",
            "target_language": "ko" (선택 사항, 기본값: "ko")
        }
    - 응답:
        {
            "status": "success",
            "translated_text": "번역된 텍스트"
        }
    """
    data = request.json
    text = data.get("text")
    target_language = data.get("target_language", "ko")

    if not text:
        return jsonify({"status": "error", "message": "번역할 텍스트가 필요합니다."}), 400

    translated_text = translate_and_filter_text(text, target_language)
    return jsonify({"status": "success", "translated_text": translated_text})

@app.route('/tts', methods=['POST'])
def tts():
    """
    텍스트를 음성으로 변환하는 API (TTS)
    - 요청 형식:
        {
            "text": "변환할 텍스트"
        }
    - 응답:
        변환된 음성 파일 다운로드
    """
    data = request.json
    text = data.get("text")


    if not text:
        return jsonify({"status": "error", "message": "텍스트 입력이 필요합니다."}), 400

    s3_url = generate_tts(text)

    if isinstance(s3_url, dict) and s3_url.get("status") == "error":
        return jsonify(s3_url), 500

    return jsonify({"status": "success", "s3_url": s3_url})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)