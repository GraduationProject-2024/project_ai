from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import configparser
config = configparser.ConfigParser()
config.read('keys.config')

TWILIO_ACCOUNT_SID = config['SMS']["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = config['SMS']["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = config['SMS']["TWILIO_PHONE_NUMBER"]

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_messages(to="+8201051321887", message=None):
    """
    Twilio를 사용하여 SMS를 전송하는 함수
    :param to: 수신자 전화번호 (예: "+821012345678")
    :param message: 전송할 메시지 내용
    :return: 전송 결과
    """
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to= to
        )
        
        ##메시지 생성 (MMS 지원)
        #msg_params = {
        #    "body": message,
        #    "from_": TWILIO_PHONE_NUMBER,
        #    "to": to
        #}

        ##이미지 처리 불가
        # #**MMS를 위한 media_url 리스트 초기화**
        #media_urls = []

        ##**이미지 업로드 후 URL 추가**
        #if image:
        #    image_url = upload_image_to_s3(image)
        #    if image_url:
        #        media_urls.append(image_url)

        ##**MMS 전송 (미디어가 있을 경우)**
        #if media_urls:
        #    msg_params["media_url"] = media_urls
        #print("media_urls: ", media_urls)
        #message = client.messages.create(**msg_params)

        return {"status": "success", "sid": message.sid}
    #용량 제한
    except Exception as e:
        return {"status": "error", "message": str(e)}