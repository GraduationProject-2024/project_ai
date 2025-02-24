import pymysql
import os
# ✅ 데이터베이스 설정
DB_CONFIG = {
    "host": os.getenv('DB_HOST'),
    "user": os.getenv('DB_ID'),
    "password": os.getenv('DB_PW'),
    "database": os.getenv('DB_DB'),
    "cursorclass": pymysql.cursors.DictCursor  # 결과를 딕셔너리 형태로 반환
}

# ✅ 언어 매핑 (0~4 값 → 언어명)
LANGUAGE_MAPPING = {
    0: "Chinese",
    1: "Vietnamese",
    2: "Mongolian",
    3: "English",
    4: "Korean"
}

# ✅ 사용자 `main_language` 조회 함수
def get_main_language(user_id):
    connection = pymysql.connect(**DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            # ✅ 한 번의 쿼리로 `member_id` 및 `main_language` 조회
            sql = """
            SELECT b.language FROM basic_info b
            JOIN member m ON m.id = b.member_id
            WHERE m.login_id = %s
            """
            cursor.execute(sql, (user_id,))
            result = cursor.fetchone()
            print(result)
            if result and "language" in result:
                return LANGUAGE_MAPPING.get(result["language"], "Unknown")  # 매핑되지 않으면 Unknown 반환
    except Exception as e:
        print(f"❌ 데이터베이스 조회 오류: {e}")
        return "Unknown"
    finally:
        connection.close()