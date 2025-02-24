from langdetect import detect

language_map = {
    "ko": "한국어",
    "en": "영어",
    "vi": "베트남어",
    "zh-cn": "중국어 간체",
    "zh-tw": "중국어 번체"
}

def detect_language(text):
    """
    입력된 텍스트의 언어를 감지하여 언어명 반환
    """
    try:
        lang_code = detect(text)
        return language_map.get(lang_code, "Unknown")  # 매핑되지 않으면 'Unknown' 반환
    except Exception as e:
        print(f"❌ 언어 감지 실패: {e}")
        return "Unknown"
