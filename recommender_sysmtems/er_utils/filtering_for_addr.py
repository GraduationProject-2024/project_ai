import xml.etree.ElementTree as ET
import configparser

class AddressFilter:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('keys.config')
        self.service_key = config['API_KEYS']['public_portal_api_key']

    def fetch_trauma_center_hpids(self):
        """
        외상센터의 모든 hpid를 수집
        """
        # 외상센터 API URL
        url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getStrmBassInfoInqire"
        params = {
            "ServiceKey": self.service_key,
            "pageNo": 1,
            "numOfRows": 100  # 페이지당 최대 항목 수
        }

        from .apis import call_api
        hpids = []

        # 첫 호출로 전체 페이지 수 확인
        xml_data = call_api(url, params)
        if not xml_data:
            return []

        root = ET.fromstring(xml_data)
        total_count = int(root.find(".//totalCount").text)  # 총 항목 수
        total_pages = (total_count + params["numOfRows"] - 1) // params["numOfRows"]  # 전체 페이지 계산

        # 모든 페이지에서 데이터 수집
        for page in range(1, total_pages + 1):
            params["pageNo"] = page
            xml_data = call_api(url, params)
            if not xml_data:
                continue
            root = ET.fromstring(xml_data)
            hpids.extend([item.find("hpid").text for item in root.findall(".//item")])

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

        try:
            root = ET.fromstring(xml_data)
            item = root.find(".//item")
            if item is not None:
                dutyAddr = item.find("dutyAddr").text if item.find("dutyAddr") is not None else None
                wgs84Lat = item.find("wgs84Lat").text if item.find("wgs84Lat") is not None else None
                wgs84Lon = item.find("wgs84Lon").text if item.find("wgs84Lon") is not None else None
                #print(dutyAddr, wgs84Lat, wgs84Lon)
                return dutyAddr, wgs84Lat, wgs84Lon
            else:
                print(f"API 응답에 item 없음: hpid={hpid}")
                return None, None, None
        except ET.ParseError as e:
            print(f"XML 파싱 오류: {e}")
            return None, None, None

    def enrich_filtered_df(self, filtered_df):
        #filtered_df에 새로운 열 추가
        trauma_hpids = self.fetch_trauma_center_hpids()

        #새로운 열 초기화
        filtered_df = filtered_df.copy()  #SettingWithCopyWarning 방지
        filtered_df["dutyAddr"] = None
        filtered_df["wgs84Lat"] = None
        filtered_df["wgs84Lon"] = None
        filtered_df["is_trauma"] = False  # 외상센터 여부를 나타내는 열 추가

        #hpid에 따라 새로운 열 값 채우기
        for index, row in filtered_df.iterrows():
            hpid = row["hpid"]
            is_trauma = hpid in trauma_hpids
            dutyAddr, wgs84Lat, wgs84Lon = self.fetch_location_data(hpid, is_trauma)
            # DataFrame에 데이터 추가
            filtered_df.at[index, "dutyAddr"] = dutyAddr
            filtered_df.at[index, "wgs84Lat"] = wgs84Lat
            filtered_df.at[index, "wgs84Lon"] = wgs84Lon
            filtered_df.at[index, "is_trauma"] = is_trauma  # 외상센터 여부 업데이트


        #dutyName 뒤에 dutyAddr, wgs84Lat, wgs84Lon 열 재배치
        columns = filtered_df.columns.tolist()
        if "dutyName" in columns:
            #dutyName의 인덱스 찾기
            duty_name_index = columns.index("dutyName")
            #새로운 열 제거
            columns.remove("dutyAddr")
            columns.remove("wgs84Lat")
            columns.remove("wgs84Lon")
            columns.remove("is_trauma")
            for col in ["dutyAddr", "wgs84Lat", "wgs84Lon", "is_trauma"]:
                columns.insert(duty_name_index + 1, col)
                duty_name_index += 1

        #DataFrame 열 재정렬
        filtered_df = filtered_df[columns]
        #print(filtered_df['wgs84Lat'], filtered_df['wgs84Lon'])
        return filtered_df