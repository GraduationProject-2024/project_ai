from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import os
# Twilio 설정
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


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

        return {"status": "success", "sid": message.sid}
    #용량 제한
    except Exception as e:
        return {"status": "error", "message": str(e)}