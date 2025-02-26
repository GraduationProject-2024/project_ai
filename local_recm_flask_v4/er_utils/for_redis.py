import redis
import configparser

def get_redis_client():
    """
    Initializes and returns a Redis client instance.
    """
    config = configparser.ConfigParser()
    config.read('keys.config')

    return redis.StrictRedis(
        host=config['EC2_INFO']['host'],
        port=6379,
        password=config['EC2_INFO']['password'],
        decode_responses=True
    )
