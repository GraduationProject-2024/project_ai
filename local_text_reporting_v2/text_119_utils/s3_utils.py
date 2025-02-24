from datetime import datetime
import configparser
config = configparser.ConfigParser()
config.read('C:/Users/user/Desktop/project_ai/keys.config')
import boto3

S3_BUCKET_NAME = config['S3_INFO']['BUCKET_NAME']

s3_client = boto3.client("s3",
                        aws_access_key_id=config['S3_INFO']['ACCESS_KEY_ID'],
                        aws_secret_access_key=config['S3_INFO']['SECRET_ACCESS_KEY'],
                        region_name="ap-northeast-2")
# AWS S3 설정
def upload_to_s3(file, folder):
    """
    파일을 AWS S3에 업로드
    :param file: 업로드할 파일 객체 (request.files['audio'])
    :param folder: S3 내 폴더 경로 (예: "uploads/")
    :return: S3 URL
    """
    
    file_name = f"{folder}{int(datetime.now().timestamp())}_{file.filename}"
    s3_client.upload_fileobj(file, S3_BUCKET_NAME, file_name)
    
    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
    print(f"파일 업로드 완료:{s3_url}")
    return s3_url


import tempfile

def download_from_s3(s3_url):
    """
    S3 URL에서 파일을 다운로드하여 로컬 파일로 저장
    :param s3_url: S3에 업로드된 파일 URL
    :return: 로컬 파일 경로
    """
    # S3 URL에서 key 추출
    key = s3_url.replace(f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/", "")

    # S3 객체 존재 여부 확인
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=key)
    except Exception as e:
        print(f"❌ S3에서 파일을 찾을 수 없음: {s3_url}")
        raise Exception("Failed to download file from S3")

    # 임시 파일 생성
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    try:
        with open(temp_audio.name, "wb") as f:
            s3_client.download_fileobj(S3_BUCKET_NAME, key, f)
    except Exception as e:
        print(f"❌ S3에서 파일 다운로드 실패: {e}")
        raise Exception("Failed to download file from S3")

    #print(f"✅ S3에서 다운로드 성공: {temp_audio.name}")
    return temp_audio.name  # 로컬 파일 경로 반환

def download_from_s3_image(s3_url):
    """
    S3 URL에서 파일을 다운로드하여 로컬 파일로 저장
    :param s3_url: S3에 업로드된 파일 URL
    :return: 로컬 파일 경로
    """
    # S3 URL에서 key 추출
    key = s3_url.replace(f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/", "")

    # S3 객체 존재 여부 확인
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=key)
    except Exception as e:
        print(f"❌ S3에서 파일을 찾을 수 없음: {s3_url}")
        raise Exception("Failed to download file from S3")

    # 임시 파일 생성
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    try:
        with open(temp_image.name, "wb") as f:
            s3_client.download_fileobj(S3_BUCKET_NAME, key, f)
    except Exception as e:
        print(f"❌ S3에서 파일 다운로드 실패: {e}")
        raise Exception("Failed to download file from S3")

    #print(f"✅ S3에서 다운로드 성공: {temp_image.name}")
    return temp_image.name  # 로컬 파일 경로 반환

def upload_image_to_s3(image_file, folder="images/"):
    """
    이미지를 AWS S3에 업로드하여 URL 반환
    :param image_file: 업로드할 이미지 파일 객체
    :return: S3에 저장된 이미지 URL
    """
    return upload_to_s3(image_file, folder)  # 기존 함수 활용