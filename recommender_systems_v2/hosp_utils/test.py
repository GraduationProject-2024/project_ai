import openai

import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

response = openai.embeddings.create(
    model="text-embedding-3-small",  # 최신 모델 사용
    input=["The food was delicious and the waiter was kind.", "I love programming in Python!"],
    encoding_format="float"  # 벡터 포맷을 float로 설정
)

# 출력
for i, embedding in enumerate(response.data):
    print(f"Text {i+1}: {embedding.embedding[:5]}... (truncated)")  # 앞부분만 출력
