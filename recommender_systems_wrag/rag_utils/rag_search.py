import openai
import json
import numpy as np
from configparser import ConfigParser
from typing import List, Dict, Any
import boto3
import io

# API 키 설정
config = ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

def load_vector_db() -> List[Dict[str, Any]]:
    """S3에서 벡터 DB JSON을 로드"""
    config = ConfigParser()
    config.read('keys.config')

    s3 = boto3.client(
        's3',
        aws_access_key_id=config['S3_INFO']['ACCESS_KEY_ID'],
        aws_secret_access_key=config['S3_INFO']['SECRET_ACCESS_KEY']
    )

    bucket = config['S3_INFO']['BUCKET_NAME2']
    key = 'medical_data/vectorized_data.json'

    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return json.loads(content)

def create_query_embedding(query: str) -> np.ndarray:
    """쿼리 텍스트의 임베딩 생성"""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar_conditions(symptoms: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    증상에 기반하여 유사한 질병 정보 검색
    
    Args:
        symptoms: 증상 정보 리스트
        top_k: 반환할 결과 수
    
    Returns:
        유사한 질병 정보 리스트
    """
    # 증상 텍스트 생성
    symptom_text = " ".join([
        f"{', '.join(s.get('macro_body_parts', []))} {', '.join(s.get('micro_body_parts', []))} {json.dumps(s.get('symptom_details', {}), ensure_ascii=False)}"
        for s in symptoms
    ])
    
    # 쿼리 임베딩 생성
    query_embedding = create_query_embedding(symptom_text)
    
    # 벡터 DB 로드
    vector_db = load_vector_db()
    
    # 유사도 계산 및 정렬
    results = []
    for item in vector_db:
        similarity = cosine_similarity(query_embedding, np.array(item["embedding"]))
        results.append({
            "disease": item["disease"],
            "symptoms": item["symptoms"],
            "similarity": float(similarity)
        })
    
    # 유사도 기준 정렬 및 상위 k개 반환
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k] 