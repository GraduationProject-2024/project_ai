DEPARTMENT_TRANSLATIONS={
    "가정의학과": {
        "KO": "가정의학과",
        "EN": "Family Medicine",
        "VI": "y học gia đình",
        "ZH_CN": "家庭医学科",
        "ZH_TW": "家庭醫學"
    },
    "내과": {
        "KO": "내과",
        "EN": "Internal Medicine",
        "VI": "khoa nội, bệnh viện nội khoa",
        "ZH_CN": "内科",
        "ZH_TW": "內科"
    },
    "마취통증의학과": {
        "KO": "마취통증의학과",
        "EN": "Anaesthesiology",
        "VI": "khoa chứng đau gây mê",
        "ZH_CN": "麻醉疼痛医学科",
        "ZH_TW": "麻醉痛医学科"
    },
    "비뇨의학과": {
        "KO": "비뇨의학과",
        "EN": "Urology",
        "VI": "khoa tiết niệu",
        "ZH_CN": "泌尿医学系",
        "ZH_TW": "泌尿学系"
    },
    "산부인과": {
        "KO": "산부인과",
        "EN": "Obstetrics and Gynecology",
        "VI": "khoa phụ sản, bệnh viện phụ sản",
        "ZH_CN": "妇产科",
        "ZH_TW": "婦產科"
    },
    "성형외과": {
        "KO": "성형외과",
        "EN": "Plastic & Reconstructive Surgery",
        "VI": "Phẫu thuật tạo hình và tái tạo",
        "ZH_CN": "整形及重建外科",
        "ZH_TW": "整形及重建外科"
    },
    "소아청소년과": {
        "KO": "소아청소년과",
        "EN": "Pediatrics",
        "VI": "khoa nhi",
        "ZH_CN": "儿童青少年科",
        "ZH_TW": "小儿青少年科"
    },
    "신경과": {
        "KO": "신경과",
        "EN": "Neurology",
        "VI": "Thần kinh học",
        "ZH_CN": "神经科",
        "ZH_TW": "神经科"
    },
    "신경외과": {
        "KO": "신경외과",
        "EN": "Neurological Surgery",
        "VI": "khoa ngoại thần kinh, bệnh viện ngoại khoa",
        "ZH_CN": "神经外科",
        "ZH_TW": "神经外科"
    },
    "심장혈관흉부외과": {
        "KO": "심장혈관흉부외과",
        "EN": "Thoracic Surgery",
        "VI": "khoa ngoại khoa tim mạch",
        "ZH_CN": "心血管胸外科",
        "ZH_TW": "心血管胸外科"
    },
    "안과": {
        "KO": "안과",
        "EN": "Ophthalmology",
        "VI": "nhãn khoa, bệnh viện mắt",
        "ZH_CN": "眼科",
        "ZH_TW": "眼科"
    },
    "영상의학과": {
        "KO": "영상의학과",
        "EN": "Imaging Radiology",
        "VI": "ngành X-quang",
        "ZH_CN": "影像医学科",
        "ZH_TW": "影像放射學"
    },
    "예방의학과": {
        "KO": "예방의학과",
        "EN": "Preventive Medicine",
        "VI": "Y học dự phòng",
        "ZH_CN": "预防医学科",
        "ZH_TW": "預防醫學"
    },
    "외과": {
        "KO": "외과",
        "EN": "General Surgery",
        "VI": "khoa ngoại, bệnh viện ngoại khoa",
        "ZH_CN": "外科",
        "ZH_TW": "一般外科"
    },
    "이비인후과": {
        "KO": "이비인후과",
        "EN": "Otolaryngology",
        "VI": "khoa tai mũi họng, bệnh viện tai mũi họng",
        "ZH_CN": "耳鼻喉科",
        "ZH_TW": "耳鼻喉科"
    },
    "재활의학과": {
        "KO": "재활의학과",
        "EN": "Rehabilitation Medicine",
        "VI": "thuốc phục hồi chức năng",
        "ZH_CN": "康复医学系",
        "ZH_TW": "康复医法系"
    },
    "정신건강의학과": {
        "KO": "정신건강의학과",
        "EN": "Psychiatry",
        "VI": "Tâm thần học",
        "ZH_CN": "心理健康医学系",
        "ZH_TW": "精神健康医学系"
    },
    "정형외과": {
        "KO": "정형외과",
        "EN": "Orthopedic Surgery",
        "VI": "khoa ngoại chỉnh hình, bệnh viện chấn thương chỉnh hình",
        "ZH_CN": "骨科手术",
        "ZH_TW": "骨科手術"
    },
    "치의과": {
        "KO": "치의과",
        "EN": "Dentistry",
        "VI": "nha khoa, bệnh viện nha khoa",
        "ZH_CN": "牙科",
        "ZH_TW": "牙科"
    },
    "피부과": {
        "KO": "피부과",
        "EN": "Dermatology",
        "VI": "khoa da liễu, bệnh viện da liễu",
        "ZH_CN": "皮肤科",
        "ZH_TW": "皮膚科"
    },
    "한방과": {
        "KO": "한방과",
        "EN": "Oriental Medicine",
        "VI": "đông y",
        "ZH_CN": "东方医学",
        "ZH_TW": "東方醫學"
    }
}
def get_department_translation(dept_ko: str, language: str) -> dict:
    translations=DEPARTMENT_TRANSLATIONS.get(dept_ko)
    if not translations:
        return {"KO": dept_ko}
    if language == "KO":
        return {"KO": translations["KO"]}
    return {
        "KO": translations["KO"],
        language: translations.get(language, "")
    }