import uuid
from flask import Flask, jsonify, request
from datetime import datetime
from mongodb_utils import get_database
from ai_utils import transcribe_audio, translate_text, summarize_text, generate_tts, detect_language
import base64
import tempfile
import os
from main_language import get_main_language
import time

#Flask 및 WebSocket 설정
app = Flask(__name__)
db = get_database()
sessions_collection = db["sessions"]

#MongoDB 인덱스 추가(쿼리 성능 개선)
sessions_collection.create_index([("created_at", -1)])

@app.route("/start_session", methods=["POST"])
def start_session():
    start_time = time.time()  #요청 시작 시간

    data = request.get_json()
    member_id = data.get("member_id") #member의 id(PK)
    
    if not member_id:
        return jsonify({"error": "id is required"}), 400

    session_id = f"session_{uuid.uuid4().hex}"  #유니크한 session_id 생성

    #사용자 main_language 조회
    main_language = get_main_language(member_id)

    #번역 대상 초기 설정(Korean, English, main_language 포함)
    initial_languages = {"Korean", "English"}
    if main_language not in initial_languages and main_language != "Unknown":
        initial_languages.add(main_language)

    sessions_collection.insert_one({
        "_id": session_id,
        "member_id": member_id,
        "transcripts": [],
        "detected_languages": list(initial_languages),
        "created_at": datetime.now(),
        "session_start_time": start_time,  #세션 시작 시간 기록
        "ended_at": None  #세션 종료 여부 추가
    })
    
    response_time = time.time() - start_time  #API 응답 시간 측정
    db["logs"].insert_one({
            "event": "start_session",
            "member_id": member_id,
            "session_id": session_id,
            "timestamp": datetime.now(),
            "response_time": response_time
        })

    return jsonify({
        "message": "Session started",
        "session_id": session_id,
        "main_language": main_language,
        "detected_languages": list(initial_languages)
    }), 201

@app.route("/audio_chunk", methods=["POST"])
def handle_audio_chunk():
    start_time = time.time()  #요청 시작 시간
    data = request.get_json()
    session_id = data.get("session_id")
    audio_base64 = data.get("audio")

    if not session_id or not audio_base64:
        return jsonify({"error": "session_id and audio are required"}), 400

    #MongoDB에서 세션 확인
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return jsonify({"error": "Session not found"}), 404

    try:
        #Base64 디코딩 후 BytesIO 변환
        audio_bytes = base64.b64decode(audio_base64)

        #Whisper가 지원하는 형식으로 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name  #Whisper에 전달할 파일 경로

        #1️.Whisper로 음성 변환 (파일 경로 사용)
        transcript = transcribe_audio(temp_audio_path)  

        #2️.번역 실행
        #기존 transcripts에서 사용된 언어 자동 감지
        previous_languages = list(set(
            [detect_language(t["original"]) for t in session.get("transcripts", [])]
        ))

        #기존 detected_languages 유지
        detected_languages = set(session.get("detected_languages", []))
        
        detected_languages.update(previous_languages)  #기존 언어 + 새로운 언어 추가
        translation_result = translate_text(transcript, previous_languages=list(detected_languages))
        translations = translation_result["translations"]

        #translate_text에서 감지한 언어 추가
        if "detected_languages" in translation_result:
            translate_detected_languages = set(translation_result["detected_languages"])
            detected_languages.update(translate_detected_languages)  #기존 리스트에 추가


        #3️.TTS 생성 (빈 문자열 또는 None 방지)
        for lang, text in translations.items():
            if text and isinstance(text, str) and text.strip():  
                translations[lang] = {
                    "text": text,
                    "tts_url": generate_tts(text, lang)
                }
            else:
                translations[lang] = {}  #MongoDB 저장 시 안전한 빈 JSON 처리

        #4️.MongoDB에 저장
        sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"detected_languages": list(detected_languages)},
             "$push": {"transcripts": {
                "original": transcript,
                "translations": translations,
                "timestamp": datetime.now()
            }}}
        )

        #임시 파일 삭제
        os.remove(temp_audio_path)

        response_time = time.time() - start_time  #API 응답 시간 측정
        db["logs"].insert_one({
            "event": "audio_chunk",
            "session_id": session_id,
            "timestamp": datetime.now(),
            "response_time": response_time,
            "detected_languages": list(detected_languages)
        })

        return jsonify({
            "message": "Audio processed",
            "session_id": session_id,
            "transcript": transcript,
            "translations": translations,
            "detected_languages": list(detected_languages)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/end_session", methods=["POST"])
def end_session():
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    session_start_time = session.get("session_start_time", None)
    session_duration = None
    if session_start_time:
        session_duration = time.time() - session_start_time  #페이지 체류 시간 측정


    db["logs"].insert_one({
        "event": "end_session",
        "session_id": session_id,
        "timestamp": datetime.now(),
        "session_duration": session_duration  #세션 지속 시간 저장
    })

    #종료 시간 기록
    sessions_collection.update_one({"_id": session_id}, {"$set": {"ended_at": datetime.now(), "session_duration": session_duration}})

    return jsonify({"message": "Session ended", "session_id": session_id}), 200


#프론트에 transcripts 띄우는 용도
@app.route("/get_transcripts/<session_id>", methods=["GET"])
def get_transcripts(session_id):
    start_time = time.time()  #API 요청 시작 시간
    session = sessions_collection.find_one({"_id": session_id})

    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    response_time = time.time() - start_time  #응답 시간 측정
    db["logs"].insert_one({
        "event": "get_transcripts",
        "session_id": session_id,
        "timestamp": datetime.now(),
        "response_time": response_time
    })
    return jsonify({"session_id": session_id, "transcripts": session.get("transcripts", [])})

#유저별로 어떤 session 있는지 최신순으로 확인하는 용도
@app.route("/get_sessions/<member_id>", methods=["GET"])
def get_sessions(member_id):
    start_time = time.time()  #요청 시작 시간
    #해당 사용자의 모든 세션을 최신순으로 가져오기
    sessions = sessions_collection.find(
        {"member_id": member_id},#, "ended_at": None},
        {"_id": 1, "created_at": 1}
    ).sort("created_at", -1)#최신순

    #데이터 변환 (JSON 직렬화 가능하도록)
    session_list = [{"session_id": s["_id"], "created_at": s["created_at"]} for s in sessions]

    if not session_list:
        return jsonify({"error": "No sessions found for this user"}), 404
    response_time = time.time() - start_time  #응답 시간 측정
    db["logs"].insert_one({
        "event": "get_sessions",
        "member_id": member_id, #"user_id": user_id,
        "timestamp": datetime.now(),
        "response_time": response_time,
        "session_count": len(session_list)  #유저의 세션 개수도 기록
    })
    #return jsonify({"user_id": user_id, "sessions": session_list})
    return jsonify({"member_id": member_id, "sessions": session_list})

#세션 기록 조회 및 요약 API
import json

@app.route("/session_summary/<session_id>", methods=["GET"])
def get_session_summary(session_id):
    start_time = time.time()  #요청 시작 시간
    session = sessions_collection.find_one({"_id": session_id})
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    full_transcript = "\n".join([t["original"] for t in session["transcripts"]])
    summary_text = summarize_text(full_transcript)

    try:
        summary = json.loads(summary_text)  #문자열이 아닌 JSON 객체로 변환
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse summary"}), 500  #JSON 파싱 실패 시 예외 처리

    response_time = time.time() - start_time  #응답 시간 측정
    db["logs"].insert_one({
        "event": "session_summary",
        "session_id": session_id,
        "timestamp": datetime.now(),
        "response_time": response_time
    })

    return jsonify(summary)  #JSON 객체 그대로 반환


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True, threaded=True)