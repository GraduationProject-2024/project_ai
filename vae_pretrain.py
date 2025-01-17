# import pandas as pd
# from hosp_utils.recommendation import HospitalRecommender
# from concurrent.futures import ThreadPoolExecutor
# from utils.direction import calculate_travel_time
# import torch

# def process_hospital_data(hospitals_df):
#     """
#     CSV에서 로드한 병원 데이터를 필터링 및 매핑합니다.

#     Parameters:
#         hospitals_df (pd.DataFrame): 병원 데이터프레임.

#     Returns:
#         pd.DataFrame: 필터링 및 매핑된 병원 데이터프레임.
#     """
#     processed_results = []
#     for _, row in hospitals_df.iterrows():
#         processed_results.append({
#             "id": row.get("id"),
#             "name": row.get("yadmNm"),
#             "address": row.get("addr"),
#             "telephone": row.get("telno"),
#             "department": row.get("dgsbjt"),
#             "latitude": row.get("yPos"),
#             "longitude": row.get("xPos"),
#             "sidocdnm": row.get("sidoCdNm"),
#             "sggucdnm": row.get("sgguCdNm"),
#             "emdongnm": row.get("emdongNm"),
#             "clcdnm": row.get("clCdNm"),
#             "location": f"{row.get('yPos')},{row.get('xPos')}",
#             "url": row.get("hospUrl")
#         })

#     return pd.DataFrame(processed_results)


# def sample_hospitals_by_region_and_type_with_travel_time(hospitals_df, user_lat, user_lon, region_col="sidocdnm", type_col="clcdnm", frac=0.1, max_samples=6000):
#     print("Starting stratified sampling...")

#     sampled_df = pd.DataFrame()  # 샘플링 결과를 저장할 데이터프레임

#     for region, region_group in hospitals_df.groupby(region_col):
#         for hosp_type, type_group in region_group.groupby(type_col):
#             # 'xPos'와 'yPos'가 null이 아닌 데이터만 필터링
#             valid_type_group = type_group.dropna(subset=['latitude', 'longitude'])
#             if valid_type_group.empty:
#                 continue  # 유효한 데이터가 없으면 다음 그룹으로 넘어감
            
#             n_samples = max(1, int(len(valid_type_group) * frac))  # 최소 1개는 샘플링
#             sampled_group = valid_type_group.sample(n=n_samples, random_state=42)
#             sampled_df = pd.concat([sampled_df, sampled_group], ignore_index=True)

#             # Sampled data size 제한
#             if len(sampled_df) >= max_samples:
#                 sampled_df = sampled_df.iloc[:max_samples]  # 초과된 경우 자름
#                 print(f"Sampled data size reached the limit: {len(sampled_df)}")
#                 break
        
#         if len(sampled_df) >= max_samples:
#             break
#     print(f"Original data size: {len(hospitals_df)}, Sampled data size: {len(sampled_df)}")

#     # 소요 시간 계산
#     print("Calculating travel times...")
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         travel_times = list(
#             executor.map(
#                 lambda row: calculate_travel_time(row, user_lat, user_lon),
#                 sampled_df.to_dict("records")
#             )
#         )

#     sampled_df['travel_time'] = travel_times
#     sampled_df['travel_time_sec'] = sampled_df['travel_time'].apply(lambda x: x % 60 if x is not None else None)
#     sampled_df['travel_time_h'] = sampled_df['travel_time'] // 3600
#     sampled_df['travel_time_min'] = (sampled_df['travel_time'] % 3600) // 60

#     print("Travel time calculation complete.")
#     return sampled_df

# # 병원 데이터 로드 및 전처리
# hospitals_df = pd.read_csv('local_recm_flask/updated_hospital_data.csv')
# hospitals_df = process_hospital_data(hospitals_df)
# print(f"Total hospitals processed: {len(hospitals_df)}")

# # yadmNm, clCdNm 기준으로 샘플링 및 소요 시간 계산
# user_lat, user_lon = 37.545179, 126.964852  # 숙명여자대학교 제1캠퍼스 정문 앞
# sampled_hospitals = sample_hospitals_by_region_and_type_with_travel_time(
#     hospitals_df, user_lat, user_lon, frac=0.1
# )

# # 샘플링된 병원 데이터 확인
# print(sampled_hospitals.head())

# # 사전학습 준비
# print("Preparing hospital embeddings for VAE pretraining...")
# recommender = HospitalRecommender()
# hospital_embeddings = recommender.embed_hospital_data(sampled_hospitals)

# # VAE 학습
# vae = recommender.train_vae(
#     hospital_embeddings,
#     input_dim=hospital_embeddings.shape[1],
#     latent_dim=64,
#     hidden_dim=128,
#     epochs=100,
#     lr=0.001
# )

# # VAE 모델 저장
# torch.save(vae.state_dict(), "vae_pretrained_model.pth")
# print("VAE model pretrained and saved successfully!")