# from langdetect import detect

# def detect_language(text):
#     """
#     입력된 텍스트의 언어를 감지
#     :param text: 감지할 텍스트
#     :return: 감지된 언어 코드 ("ko", "en", "vi", etc.)
#     """
#     try:
#         lang = detect(text)
#         return lang
#     except Exception as e:
#         print(f"❌ 언어 감지 실패: {e}")
#         return "unknown"  # 감지 실패 시 기본값 설정