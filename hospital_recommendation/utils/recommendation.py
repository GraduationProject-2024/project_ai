from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from utils.basic_autoencoder import AutoEncoder

class HospitalRecommender:
    def __init__(self, model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        #https://brunch.co.kr/@b2439ea8fc654b8/70 참고해서 모델 결정(경량화된 모델)

    def train_autoencoder(self, data, input_dim, latent_dim, epochs=50, lr=0.001):
        """
        AutoEncoder 학습
        """
        model = AutoEncoder(input_dim, latent_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        tensor_data = torch.tensor(data, dtype=torch.float32)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            latent, reconstructed = model(tensor_data)
            loss = criterion(reconstructed, tensor_data)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
        return model

    #사용자 프로필 임베딩 함수
    def embed_user_profile(self, basic_info, health_info):
        text_data = " ".join([
            health_info["pastHistory"] or "",
            health_info["familyHistory"] or "",
            health_info["nowMedicine"] or "",
            health_info["allergy"] or ""
        ])
        text_embedding = self.embedding_model.encode([text_data])

        numeric_features = np.array([basic_info["age"], basic_info["height"], basic_info["weight"]])
        norm = np.linalg.norm(numeric_features)
        numeric_embedding = numeric_features / norm if norm != 0 else numeric_features

        
        return np.hstack((text_embedding[0], numeric_embedding))

    #병원 데이터 임베딩 함수
    def embed_hospital_data(self, hospitals_df, suspected_disease=None):
        #진료과 텍스트 데이터 벡터화
        department_embeddings = self.embedding_model.encode(hospitals_df["department"].fillna("").tolist())

        #병원 유형 텍스트 임베딩
        clcdnm_embeddings = self.embedding_model.encode(hospitals_df["clcdnm"].fillna("").tolist())

        #병원 이름 텍스트 임베딩 추가
        name_embeddings = self.embedding_model.encode(hospitals_df["name"].fillna("").tolist())

        #병원 소요 시간 데이터 정규화
        scaler = StandardScaler()
        time_features = hospitals_df[["travel_time_h", "travel_time_min"]].fillna(0).values
        time_embedding = scaler.fit_transform(time_features)

        #의심 질병 임베딩(선택)
        if suspected_disease:
            suspected_disease_embedding = self.embedding_model.encode([suspected_disease])[0]
            suspected_disease_embeddings = np.tile(suspected_disease_embedding, (hospitals_df.shape[0], 1))
        else:
            suspected_disease_embeddings = np.zeros((hospitals_df.shape[0], department_embeddings.shape[1]))

        #최종 병원 벡터 결합
        return np.hstack((department_embeddings, clcdnm_embeddings, name_embeddings, suspected_disease_embeddings, time_embedding))


    #추천 함수
    def recommend_hospitals(self, autoencoder, user_embedding, hospital_embeddings, hospitals_df, department=None, suspected_disease=None, use_autoencoder=False):
        #사용자 임베딩 차원 맞추기 (Zero Padding)
        if user_embedding.shape[0] < hospital_embeddings.shape[1]:
            padding = hospital_embeddings.shape[1] - user_embedding.shape[0]
            user_embedding_padded = np.pad(user_embedding, (0, padding))
        else:
            user_embedding_padded = user_embedding

        if use_autoencoder:
            #AutoEncoder를 사용하여 latent space로 변환
            user_tensor = torch.tensor(user_embedding_padded, dtype=torch.float32).unsqueeze(0)
            hospital_tensor = torch.tensor(hospital_embeddings, dtype=torch.float32)

            user_latent, _ = autoencoder(user_tensor)
            hospital_latents, _ = autoencoder(hospital_tensor)

            #코사인 유사도 계산
            similarities = cosine_similarity(user_latent.detach().numpy(), hospital_latents.detach().numpy())
        else:
            # AutoEncoder를 사용하지 않고 코사인 유사도 직접 계산
            similarities = cosine_similarity([user_embedding_padded], hospital_embeddings)

        #유사도 결과를 병원 데이터프레임에 추가
        hospitals_df["similarity"] = similarities[0]

        #department 유사도 추가(선택)
        if department:
            #병원 이름에 department 포함 여부 확인 및 가중치 부여
            hospitals_df["department_match"] = hospitals_df["name"].apply(lambda name: department in name)
            hospitals_df["similarity"] += hospitals_df["department_match"] * 0.1  # 가중치 0.1 추가    

        #suspected_disease와 병원 이름 유사도 추가(선택)
        if suspected_disease:
            disease_embeddings = self.embedding_model.encode([suspected_disease])
            hospital_name_embeddings = self.embedding_model.encode(hospitals_df["name"].fillna("").tolist())
            disease_similarities = cosine_similarity(disease_embeddings, hospital_name_embeddings)[0]
            hospitals_df["similarity"] += disease_similarities * 0.001  # 가중치 0.05 추가

        #similarity 값이 1을 초과하지 않도록 제한
        hospitals_df["similarity"] = hospitals_df["similarity"].clip(upper=1.0)
        #유사도 기준으로 정렬
        recommended = hospitals_df.sort_values(by="similarity", ascending=False)
        recommended = recommended.reset_index(drop=True)
        return recommended