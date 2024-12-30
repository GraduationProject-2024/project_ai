import configparser
import xml.etree.ElementTree as ET

class AddressFilter:
    def __init__(self, config_path):
        #ConfigParser 초기화 및 설정 파일 읽기
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.service_key = self.config['API_KEYS']['public_portal_api_key']

    def fetch_trauma_center_hpids(self):
        #외상센터 hpid 수집
        url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getStrmBassInfoInqire"
        params = {"ServiceKey": self.service_key}

        from .apis import call_api
        xml_data = call_api(url, params)
        if not xml_data:
            return []

        root = ET.fromstring(xml_data)
        hpids = [item.find("hpid").text for item in root.findall(".//item")]
        return hpids

    def fetch_location_data(self, hpid, is_trauma):
        #지정된 hpid에 대해 dutyAddr, wgs84Lat, wgs84Lon 정보 수집
        base_url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/"
        endpoint = "getStrmBassInfoInqire" if is_trauma else "getEgytBassInfoInqire"
        url = f"{base_url}{endpoint}"
        params = {"ServiceKey": self.service_key, "HPID": hpid}

        from .apis import call_api
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

    def enrich_filtered_df(self, filtered_df):
        #filtered_df에 새로운 열 추가
        trauma_hpids = self.fetch_trauma_center_hpids()
        print(f"외상센터 hpid 수집 완료: {trauma_hpids}")

        #새로운 열 초기화
        filtered_df = filtered_df.copy()  #SettingWithCopyWarning 방지
        filtered_df["dutyAddr"] = None
        filtered_df["wgs84Lat"] = None
        filtered_df["wgs84Lon"] = None

        #hpid에 따라 새로운 열 값 채우기
        for index, row in filtered_df.iterrows():
            hpid = row["hpid"]
            is_trauma = hpid in trauma_hpids
            dutyAddr, wgs84Lat, wgs84Lon = self.fetch_location_data(hpid, is_trauma)
            filtered_df.at[index, "dutyAddr"] = dutyAddr
            filtered_df.at[index, "wgs84Lat"] = wgs84Lat
            filtered_df.at[index, "wgs84Lon"] = wgs84Lon

        #dutyName 뒤에 dutyAddr, wgs84Lat, wgs84Lon 열 재배치
        columns = filtered_df.columns.tolist()
        if "dutyName" in columns:
            #dutyName의 인덱스 찾기
            duty_name_index = columns.index("dutyName")
            #새로운 열 제거
            columns.remove("dutyAddr")
            columns.remove("wgs84Lat")
            columns.remove("wgs84Lon")
            #새로운 열을 dutyName 뒤에 삽입
            for col in ["dutyAddr", "wgs84Lat", "wgs84Lon"]:
                columns.insert(duty_name_index + 1, col)
                duty_name_index += 1

        #DataFrame 열 재정렬
        filtered_df = filtered_df[columns]

        return filtered_df