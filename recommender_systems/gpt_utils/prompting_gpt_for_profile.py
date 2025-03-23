import openai
import configparser

config = configparser.ConfigParser()
config.read('keys.config')
openai.api_key = config['API_KEYS']['chatgpt_api_key']

def translate_text(text, target_language="en"):
    """
    사용자가 입력한 텍스트를 target_language로 번역
    욕설이나 공격적인 표현은 정제하고, 번역이 이미 되어 있다면 그대로 유지
    """
    language_map = {
        "ko": "Korean",
        "en": "English",
        "vi": "Vietnamese",
        "zh": "Chinese(Simplified)",
        "zh-hant": "Chinese(Traditional)"
    }

    target_lang_full = language_map.get(target_language.lower(), "English")

    prompt = (
        f"You are a professional medical translator. "
        f"Translate the given short text to {target_lang_full} if it is not already in that language. "
        f"Do NOT explain what language the input is in. "
        f"DO NOT include any introductory phrases such as 'Here is the translation' or 'The input means...'. "
        f"Only return the translated result — no commentary, no explanation. "
        f"If the text is already in {target_lang_full}, return it as is (cleaned). "
        f"If the text contains any vulgar or offensive expressions, rewrite them politely. "
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT translation error]: {e}")
        return text  #fallback: 원본 반환
