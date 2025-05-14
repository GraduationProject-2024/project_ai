import pymongo
import configparser

def get_mongo_client():
    config = configparser.ConfigParser()
    config.read('keys.config')
    
    user = config['MONGODB_INFO']['user']
    password = config['MONGODB_INFO']['password']
    cluster_url = config['MONGODB_INFO']['cluster_url']
    
    client = pymongo.MongoClient(
        f"mongodb+srv://{user}:{password}@{cluster_url}/?retryWrites=true&w=majority&appName=mediko-free"
    )
    return client

def get_database():
    client = get_mongo_client()
    return client["audio_transcription"]