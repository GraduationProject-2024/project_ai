import pymongo
import os

def get_mongo_client():

    cluster_url = os.getenv('MONGODB_CLUSTER_URL')
    user = os.getenv('MONGODB_USER')
    password = os.getenv('MONGODB_PW')
    
    client = pymongo.MongoClient(
        f"mongodb+srv://{user}:{password}@{cluster_url}/?retryWrites=true&w=majority&appName=mediko-free"
    )
    return client

def get_database():
    client = get_mongo_client()
    return client["audio_transcription"]

