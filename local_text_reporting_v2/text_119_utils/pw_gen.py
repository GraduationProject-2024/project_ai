import string
import random
import pymysql

import configparser
# API 키 설정
config = configparser.ConfigParser()
config.read('C:/Users/user/Desktop/project_ai/keys.config')


# MySQL 연결 설정
DB_CONFIG = {
    "host": config['DB_INFO']['host'],
    "user": config['DB_INFO']['id'],
    "password": config['DB_INFO']['password'],
    "database": config['DB_INFO']['db'],
    "cursorclass": pymysql.cursors.DictCursor  # 딕셔너리 형태로 결과 반환
}

def get_used_passwords():
    """MySQL에서 사용된 비밀번호 목록 조회 (pymysql 사용)"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            cursor.execute("SELECT used_passwords FROM password_table;")
            result = cursor.fetchall()
        conn.close()
        return [row["used_passwords"] for row in result]  # 리스트로 변환
    except pymysql.MySQLError as err:
        print(f"MySQL Error: {err}")
        return []
    
def generate_password(used_passwords):
    characters = string.ascii_letters + string.digits  # 알파벳 대소문자 + 숫자 포함
    while True:
        length = random.randint(6, 16)  # 6~16자리 랜덤 길이 선택
        password = ''.join(random.choices(characters, k=length))
        if password not in used_passwords and not has_continuous_sequence(password):
            return password

def has_continuous_sequence(password):
    for i in range(len(password) - 2):
        seq = password[i:i+3]
        if is_continuous(seq):
            return True
    return False

def is_continuous(seq):
    return seq in string.ascii_lowercase or seq in string.ascii_uppercase or seq in string.digits or \
           seq[::-1] in string.ascii_lowercase or seq[::-1] in string.ascii_uppercase or seq[::-1] in string.digits
