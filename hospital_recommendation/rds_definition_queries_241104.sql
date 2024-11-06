-- Step 1: Database 생성
CREATE DATABASE IF NOT EXISTS hospitals;

-- Step 2: Database 사용
USE hospitals;

-- Step 3: Table 생성
CREATE TABLE IF NOT EXISTS hospital_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    addr VARCHAR(255) NOT NULL,             -- 병원 주소
    yadmNm VARCHAR(255) NOT NULL,           -- 병원 이름
    sidoCd VARCHAR(10),                     -- 시도 코드
    sgguCd VARCHAR(10),                     -- 시군구 코드
    xPos DECIMAL(10, 7),                    -- 병원 위치 x 좌표 (경도)
    yPos DECIMAL(10, 7),                    -- 병원 위치 y 좌표 (위도)
    dgsgbjtCd VARCHAR(10),                  -- 진료과목 코드
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
