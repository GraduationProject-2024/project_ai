def clean_form_value(val):
    if val is None:
        return None
    val = str(val).strip().lower()
    val = val.strip("'\"")  # 양쪽 따옴표 제거
    if val in ("", "null", "none"):
        return None
    return val