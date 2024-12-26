# 1. 외상센터 hpid 수집
def fetch_trauma_center_hpids():
    import configparser
    # ConfigParser 초기화
    config = configparser.ConfigParser()

    # keys.config 파일 읽기
    config.read('C:/Users/user/Desktop/24-2/졸업프로젝트/project_ai/keys.config')

    url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getStrmBassInfoInqire"
    service_key = config['API_KEYS']['public_portal_api_key']
    params = {"ServiceKey": service_key}
    
    xml_data = call_api(url, params)
    if not xml_data:
        return []

    root = ET.fromstring(xml_data)
    hpids = [item.find("hpid").text for item in root.findall(".//item")]
    return hpids

# 2. dutyAddr, wgs84Lat, wgs84Lon 수집 및 추가
def fetch_location_data(hpid, is_trauma):
    base_url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/"
    endpoint = "getStrmBassInfoInqire" if is_trauma else "getEgytBassInfoInqire"
    url = f"{base_url}{endpoint}"
    service_key = config['API_KEYS']['public_portal_api_key']
    params = {"ServiceKey": service_key, "HPID": hpid}
    
    xml_data = call_api(url, params)
    if not xml_data:
        return None, None, None

    root = ET.fromstring(xml_data)
    item = root.find(".//item")
    if item is not None:
        dutyAddr = item.find("dutyAddr").text if item.find("dutyAddr") is not None else None
        wgs84Lat = item.find("wgs84Lat").text if item.find("wgs84Lat") is not None else None
        wgs84Lon = item.find("wgs84Lon").text if item.find("wgs84Lon") is not None else None
        return dutyAddr, wgs84Lat, wgs84Lon
    return None, None, None

# 3. filtered_df에 값 추가
def enrich_filtered_df(filtered_df):
    trauma_hpids = fetch_trauma_center_hpids()
    print(f"외상센터 hpid 수집 완료: {trauma_hpids}")

    # Initialize new columns with None
    filtered_df = filtered_df.copy()  # Avoid SettingWithCopyWarning
    filtered_df["dutyAddr"] = None
    filtered_df["wgs84Lat"] = None
    filtered_df["wgs84Lon"] = None

    # Populate the new columns based on hpid
    for index, row in filtered_df.iterrows():
        hpid = row["hpid"]
        is_trauma = hpid in trauma_hpids
        dutyAddr, wgs84Lat, wgs84Lon = fetch_location_data(hpid, is_trauma)
        filtered_df.at[index, "dutyAddr"] = dutyAddr
        filtered_df.at[index, "wgs84Lat"] = wgs84Lat
        filtered_df.at[index, "wgs84Lon"] = wgs84Lon

    # Rearrange columns to move dutyAddr, wgs84Lat, wgs84Lon after dutyName
    columns = filtered_df.columns.tolist()
    if "dutyName" in columns:
        # Find the index of dutyName
        duty_name_index = columns.index("dutyName")
        # Remove the new columns if they exist in the list
        columns.remove("dutyAddr")
        columns.remove("wgs84Lat")
        columns.remove("wgs84Lon")
        # Insert the new columns after dutyName
        for col in ["dutyAddr", "wgs84Lat", "wgs84Lon"]:
            columns.insert(duty_name_index + 1, col)
            duty_name_index += 1

    # Rearrange DataFrame
    filtered_df = filtered_df[columns]

    return filtered_df
