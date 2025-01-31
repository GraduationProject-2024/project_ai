import redis
import os

def get_redis_client():
    """
    Initializes and returns a Redis client instance.
    """
    
    return redis.StrictRedis(
        host=os.getenv("EC2_HOST"),
        port=6379,
        password=os.getenv("EC2_PW"),
        decode_responses=True
    )