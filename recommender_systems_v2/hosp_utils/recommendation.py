#from sentence_transformers import SentenceTransformer
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import numpy as np
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics.pairwise import cosine_similarity

#class HospitalRecommender:
#    def __init__(self, model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'):
#        self.embedding_model = SentenceTransformer(model_name)
#        #https://brunch.co.kr/@b2439ea8fc654b8/70 참고해서 모델 결정(경량화된 모델)

    
#    def train_vae(self, data, input_dim, hidden_dim, latent_dim, epochs=100, lr=0.001):
#        """
#        Variational Autoencoder (VAE) 학습
#        :param data: 입력 데이터 (numpy array)
#        :param input_dim: 입력 데이터 차원
#        :param hidden_dim: 중간층 차원
#        :param latent_dim: 잠재 공간 차원
#        :param epochs: 학습 반복 횟수
#        :param lr: 학습률
#        :return: 학습된 VAE 모델
#        """
#        #VAE 모델 정의
#        class VAE(nn.Module):
#            def __init__(self, input_dim, hidden_dim, latent_dim):
#                super(VAE, self).__init__()
#                self.encoder = nn.Sequential(
#                    nn.Linear(input_dim, hidden_dim),
#                    nn.LeakyReLU()
#                )
#                self.mu_layer = nn.Linear(hidden_dim, latent_dim)
#                self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
#                self.decoder = nn.Sequential(
#                    nn.Linear(latent_dim, hidden_dim),
#                    nn.LeakyReLU(),
#                    nn.Linear(hidden_dim, input_dim),
#                    nn.Sigmoid()
#                )

#            def encode(self, x):
#                h = self.encoder(x)
#                mu = self.mu_layer(h)
#                log_var = self.log_var_layer(h)
#                return mu, log_var

#            def reparameterize(self, mu, log_var):
#                std = torch.exp(0.5 * log_var)
#                eps = torch.randn_like(std)
#                return mu + eps * std

#            def decode(self, z):
#                return self.decoder(z)

#            def forward(self, x):
#                mu, log_var = self.encode(x)
#                z = self.reparameterize(mu, log_var)
#                reconstructed = self.decode(z)
#                return reconstructed, mu, log_var

#        #VAE 모델 초기화
#        vae_model = VAE(input_dim, hidden_dim, latent_dim)
#        optimizer = optim.Adam(vae_model.parameters(), lr=lr)
#        criterion = nn.MSELoss(reduction='sum')  #Reconstruction Loss

#        #데이터를 PyTorch Tensor로 변환
#        tensor_data = torch.tensor(data, dtype=torch.float32)

#        #학습 루프
#        for epoch in range(epochs):
#            vae_model.train()
#            optimizer.zero_grad()

#            reconstructed, mu, log_var = vae_model(tensor_data)
#            #손실 계산: Reconstruction Loss + KL Divergence
#            recon_loss = criterion(reconstructed, tensor_data)
#            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#            loss = recon_loss + kl_div

#            loss.backward()
#            optimizer.step()

#            #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Recon Loss: {recon_loss.item()}, KL Div: {kl_div.item()}")

#        return vae_model

#    #사용자 프로필 임베딩 함수
#    def embed_user_profile(self, basic_info, health_info):
#        text_data = " ".join([
#            health_info["pastHistory"] or "",
#            health_info["familyHistory"] or "",
#            health_info["nowMedicine"] or "",
#            health_info["allergy"] or ""
#        ])
#        text_embedding = self.embedding_model.encode([text_data])

#        numeric_features = np.array([basic_info["age"], basic_info["height"], basic_info["weight"]])
#        norm = np.linalg.norm(numeric_features)
#        numeric_embedding = numeric_features / norm if norm != 0 else numeric_features

        
#        return np.hstack((text_embedding[0], numeric_embedding))

#    #병원 데이터 임베딩 함수
#    def embed_hospital_data(self, hospitals_df, suspected_disease=None):
#        #진료과 텍스트 데이터 벡터화
#        department_embeddings = self.embedding_model.encode(hospitals_df["department"].fillna("").tolist())

#        #병원 유형 텍스트 임베딩
#        clcdnm_embeddings = self.embedding_model.encode(hospitals_df["clcdnm"].fillna("").tolist())

#        #병원 이름 텍스트 임베딩 추가
#        name_embeddings = self.embedding_model.encode(hospitals_df["name"].fillna("").tolist())

#        #병원 소요 시간 데이터 정규화
#        scaler = StandardScaler()
#        time_distance_features = hospitals_df[[
#            "transit_travel_time_h",
#            "transit_travel_time_m",
#            "transit_travel_time_s",
#            "transit_travel_distance_km"
#        ]].fillna(0).values
#        time_distance_embeddings = scaler.fit_transform(time_distance_features)

#        #의심 질병 임베딩(선택)
#        #if suspected_disease:
#        #   suspected_disease_embedding = self.embedding_model.encode([suspected_disease])[0]
#        #   suspected_disease_embeddings = np.tile(suspected_disease_embedding, (hospitals_df.shape[0], 1))
#        #else:
#        #   suspected_disease_embeddings = np.zeros((hospitals_df.shape[0], department_embeddings.shape[1]))
#        #suspected_disease를 리스트로 강제 변환
#        if suspected_disease:
#            if isinstance(suspected_disease, str):
#                suspected_disease = [suspected_disease]  #단일 값일 경우 리스트로 변환

#            #여러 의심 질병을 벡터화하고 평균값 사용
#            suspected_disease_embeddings = self.embedding_model.encode(suspected_disease)
#            avg_suspected_embedding = np.mean(suspected_disease_embeddings, axis=0)

#            suspected_disease_embeddings = np.tile(avg_suspected_embedding, (hospitals_df.shape[0], 1))
#        else:
#            suspected_disease_embeddings = np.zeros((hospitals_df.shape[0], department_embeddings.shape[1]))


#        #최종 병원 벡터 결합
#        return np.hstack((department_embeddings, 
#                          clcdnm_embeddings, 
#                          name_embeddings, 
#                          suspected_disease_embeddings, 
#                          time_distance_embeddings))

import openai
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

import configparser
config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

class HospitalRecommender:
    def __init__(self, model_name='text-embedding-3-small'):
        self.model_name = model_name  # OpenAI Embedding Model 사용

    def train_vae(self, data, input_dim, hidden_dim, latent_dim, epochs=100, lr=0.001):
        """
        Variational Autoencoder (VAE) 학습
        :param data: 입력 데이터 (numpy array)
        :param input_dim: 입력 데이터 차원
        :param hidden_dim: 중간층 차원
        :param latent_dim: 잠재 공간 차원
        :param epochs: 학습 반복 횟수
        :param lr: 학습률
        :return: 학습된 VAE 모델
        """

        class VAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super(VAE, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU()
                )
                self.mu_layer = nn.Linear(hidden_dim, latent_dim)
                self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                )

            def encode(self, x):
                h = self.encoder(x)
                mu = self.mu_layer(h)
                log_var = self.log_var_layer(h)
                return mu, log_var

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                reconstructed = self.decode(z)
                return reconstructed, mu, log_var

        # VAE 모델 초기화
        vae_model = VAE(input_dim, hidden_dim, latent_dim)
        optimizer = optim.Adam(vae_model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='sum')  # Reconstruction Loss

        # 데이터를 PyTorch Tensor로 변환
        tensor_data = torch.tensor(data, dtype=torch.float32)


            # 학습 루프
        for epoch in range(epochs):
            vae_model.train()
            optimizer.zero_grad()

            reconstructed, mu, log_var = vae_model(tensor_data)
            recon_loss = criterion(reconstructed, tensor_data)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_div

            loss.backward()
            optimizer.step()

        return vae_model
        
    def get_embedding(self, text_list, batch_size=256):
        """
        OpenAI API를 이용하여 텍스트 임베딩을 생성
        :param text_list: 리스트 형식의 입력 텍스트 데이터
        :return: numpy array 형태의 임베딩 결과
        """
        # if not isinstance(texts, list):  
        #     texts = [str(texts)]  #단일 문자열을 리스트로 변환
        # else:
        #     texts = [str(text).strip() for text in texts if text and isinstance(text, str)]  #빈 문자열 제거

        # if not texts:  #비어 있는 경우 기본값 제공
        #     texts = ["unknown"]

        # # **디버깅: 입력 데이터 확인**
        # print(f"🔹 OpenAI API 호출 - 입력 데이터 확인: {texts[:5]}")  # 처음 5개만 출력
        # print(f"🔹 입력 데이터 길이: {len(texts)}")
        
        # try:
        #     response = openai.embeddings.create(
        #         model=self.model_name,
        #         input=texts,
        #         encoding_format="float"  # float 형식으로 벡터 반환
        #     )
        #     # **디버깅: 응답 데이터 확인**
        #     print(f"✅ OpenAI API 응답 길이: {len(response.data)}")
        #     return np.array([embedding.embedding for embedding in response.data])
        # except openai.BadRequestError as e:
        #     print(f"❌ OpenAI API 요청 실패: {e}")
        #     return np.zeros((len(texts), 1536))  # 기본 Embedding 크기로 빈 벡터 반환 (1536은 OpenAI ada-002의 임베딩 차원)
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            try:
                response = openai.embeddings.create(
                    model=self.model_name,  # 최신 모델 사용 추천
                    input=batch,
                    encoding_format="float"
                )
                #print("✅ OpenAI API 응답 성공:", response)
                embeddings.extend([embedding.embedding for embedding in response.data])

            except openai.OpenAIError as e:
                #print(f"❌ OpenAI API 요청 실패: {e}")
                # 요청 실패 시, 빈 벡터 반환 (1536은 text-embedding-ada-002 모델의 기본 차원)
                zero_vector = np.zeros((len(batch), 1536))
                embeddings.extend(zero_vector)

        return np.array(embeddings)

    def embed_user_profile(self, basic_info, health_info):
        """
        사용자 건강 정보 및 기초 정보를 OpenAI Embedding을 활용하여 벡터화
        """
        text_data = [
            health_info.get("pastHistory", "unknown"),
            health_info.get("familyHistory", "unknown"),
            health_info.get("nowMedicine", "unknown"),
            health_info.get("allergy", "unknown")
        ]
        #OpenAI API는 2048자 제한 → 각 항목을 개별적으로 검사하여 잘라줌
        #평균 임베딩 계산 (리스트의 모든 벡터 평균)
        text_data = [text[:512] for text in text_data]  #문장이 너무 길면 512자로 제한

        text_embedding = self.get_embedding([text_data])  #단일 문장
        text_embedding = np.mean(text_embedding, axis=0)

        #나이, 키, 몸무게 정규화 후 결합
        # numeric_features = np.array([basic_info["age"], basic_info["height"], basic_info["weight"]])
        numeric_features = np.array([
            basic_info.get("age", 0), 
            basic_info.get("height", 0), 
            basic_info.get("weight", 0)
        ])
        norm = np.linalg.norm(numeric_features)
        numeric_embedding = numeric_features / norm if norm != 0 else numeric_features

        return np.hstack((text_embedding, numeric_embedding))

    def embed_hospital_data(self, hospitals_df, suspected_disease=None):
        """
        병원 데이터를 OpenAI API 임베딩으로 변환
        """
        for col in hospitals_df.columns:
            if hospitals_df[col].dtype == "object":
                #print("object 칼럼:", hospitals_df[col])
                hospitals_df[col] = hospitals_df[col].fillna("unknown").replace("", "unknown")
            elif hospitals_df[col].dtype in ["float64", "int64"]:
                #print("numberic 칼럼:", hospitals_df[col])
                hospitals_df[col] = hospitals_df[col].fillna(0)
                
        # 모든 기본값을 "unknown"으로 설정
        # hospital_texts = hospitals_df["name"].tolist()
        # department_texts = hospitals_df["department"].tolist()
        # clcdnm_texts = hospitals_df["clcdnm"].tolist()

        # # 🔹 OpenAI API에 전달할 텍스트 데이터 준비
        # hospital_texts = [" | ".join(text.split(",")) for text in hospitals_df["name"].tolist()]  # ✅ 쉼표 제거
        # department_texts = [" | ".join(text.split(",")) for text in hospitals_df["department"].tolist()]  # ✅ 쉼표 제거
        # clcdnm_texts = hospitals_df["clcdnm"].tolist()

        # #OpenAI API 호출을 통해 벡터화 (Batch 요청)
        # hospital_embeddings = self.get_embedding(hospital_texts)
        # department_embeddings = self.get_embedding(department_texts)
        # clcdnm_embeddings = self.get_embedding(clcdnm_texts)
        
        #병원 데이터를 하나의 문장으로 결합하여 OpenAI API로 벡터화

        hospital_sentences = hospitals_df.apply(
            lambda row: f"{row['name']}은(는) {row['sidocdnm']} {row['sggucdnm']}에 위치한 {row['clcdnm']}입니다. "
                        f"진료 과목으로는 {row['department']}이(가) 있으며, 주소는 {row['address']}입니다.",
            axis=1
        ).tolist()

        # OpenAI API 호출 (배치 단위로 처리)
        hospital_embeddings = self.get_embedding(hospital_sentences, batch_size=256)

        #시간 및 거리 정보 정규화
        scaler = StandardScaler()
        time_distance_features = hospitals_df[[
            "transit_travel_time_h",
            "transit_travel_time_m",
            "transit_travel_time_s",
            "transit_travel_distance_km"
        ]].fillna(0).values
        time_distance_embeddings = scaler.fit_transform(time_distance_features)

        #의심 질병 처리
        if suspected_disease:
            if isinstance(suspected_disease, str):
                suspected_disease = [suspected_disease]

            suspected_disease_embeddings = self.get_embedding(suspected_disease, batch_size=16)
            avg_suspected_embedding = np.mean(suspected_disease_embeddings, axis=0)
            suspected_disease_embeddings = np.tile(avg_suspected_embedding, (hospitals_df.shape[0], 1))
        else:
            suspected_disease_embeddings = np.zeros((hospitals_df.shape[0], hospital_embeddings.shape[1]))

        #최종 병원 벡터 결합
        return np.hstack((hospital_embeddings, suspected_disease_embeddings, time_distance_embeddings))

    #추천 함수
    def recommend_hospitals(self, user_embedding, hospital_embeddings, hospitals_df, vae=None, department=None, suspected_disease=None, use_vae=False):
        #사용자 임베딩 차원 맞추기 (Zero Padding)
        if user_embedding.shape[0] < hospital_embeddings.shape[1]:
            padding = hospital_embeddings.shape[1] - user_embedding.shape[0]
            user_embedding_padded = np.pad(user_embedding, (0, padding))
        else:
            user_embedding_padded = user_embedding

        if use_vae:
            #vae 사용해서 latent space로 변환
            user_tensor = torch.tensor(user_embedding_padded, dtype=torch.float32).unsqueeze(0)
            hospital_tensor = torch.tensor(hospital_embeddings, dtype=torch.float32)

            user_latent, _ = vae.encode(user_tensor)#autoencoder(user_tensor)
            hospital_latents, _ = vae.encode(hospital_tensor)#autoencoder(hospital_tensor)

            #코사인 유사도 계산
            similarities = cosine_similarity(user_latent.detach().numpy(), hospital_latents.detach().numpy())
        else:
            #vae를 사용하지 않고 코사인 유사도 직접 계산
            similarities = cosine_similarity([user_embedding_padded], hospital_embeddings)

        #유사도 결과를 병원 데이터프레임에 추가
        hospitals_df["similarity"] = similarities[0]

        #department 유사도 추가(선택)
        # if department:
        #     if department == "치의과":
        #         hospitals_df["department_match"] = hospitals_df["name"].apply(lambda name: "치과" in name if name else False)
        #     elif department == "한방과":
        #         hospitals_df["department_match"] = hospitals_df["name"].apply(lambda name: any(x in name for x in ["한의원", "한방"]) if name else False)
        #     else:
        #         hospitals_df["department_match"] = hospitals_df["name"].apply(lambda name: department in name if name else False)
            
        # #가중치 부여
        # hospitals_df["similarity"] += hospitals_df["department_match"] * 0.00001  #가중치 0.00001 추가

        #suspected_disease를 리스트로 강제 변환
        if suspected_disease:
            if isinstance(suspected_disease, str):
                suspected_disease = [suspected_disease]  #단일 값일 경우 리스트로 변환

            #여러 의심 질병을 벡터화하고 평균값 사용
            disease_embeddings = self.get_embedding(suspected_disease)
            avg_disease_embedding = np.mean(disease_embeddings, axis=0)

            hospital_name_embeddings = self.get_embedding(hospitals_df["name"].fillna("unknown").tolist())
            disease_similarities = cosine_similarity([avg_disease_embedding], hospital_name_embeddings)[0]

            hospitals_df["similarity"] += disease_similarities * 0.0001  #가중치 추가
            
        #similarity 값이 1을 초과하지 않도록 제한
        hospitals_df["similarity"] = hospitals_df["similarity"].clip(upper=1.0)
        #유사도 기준으로 정렬
        #recommended = hospitals_df.sort_values(by="similarity", ascending=False)
        #recommended = recommended.reset_index(drop=True)
        return recommended