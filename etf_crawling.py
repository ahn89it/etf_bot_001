import FinanceDataReader as fdr
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Date
import pymysql

# 1. DB 접속 정보 설정
# ----------------------------------------
db_user = 'root'
db_pass = '1234'
db_host = 'localhost'
db_port = '3306'
db_schema = 'stock'
table_name = 'kr_etf'

# DB 연결 엔진 생성
db_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_schema}?charset=utf8mb4"
engine = create_engine(db_url)
conn = engine.connect()

print(f"[INFO] DB 연결 성공: {db_host}/{db_schema}")

# 2. 수집할 종목 리스트 정의
# ----------------------------------------
target_etfs = [
    ('2006-06-27', 'KODEX 반도체', '091160'),
    ('2021-07-30', 'KODEX AI반도체', '395160'),
    ('2018-09-12', 'KODEX 2차전지산업', '305720'),
    ('2023-01-05', 'PLUS K방산', '449450'),
    ('2019-07-19', 'TIGER 리츠부동산인프라', '329200'),
    ('2024-10-22', 'TIGER 조선TOP10', '494670'),
    ('2022-11-15', 'KODEX 로봇액티브', '445290'),
    ('2011-04-06', 'TIGER 200 IT', '139260'),
    ('2006-06-27', 'KODEX 자동차', '091180'),
    ('2006-06-27', 'KODEX 은행', '091170'),
    ('2008-05-29', 'KODEX 증권', '102970'),
    ('2016-05-13', 'KODEX 바이오', '244580')
]

# 3. 데이터 수집 및 DB 저장 루프
# ----------------------------------------
print(f"[INFO] 총 {len(target_etfs)}개 종목 데이터 수집을 시작합니다.")

try:
    for start_date, name, code in target_etfs:
        print(f" -> 수집 중: {name} ({code}) ... ", end='')
        
        # FinanceDataReader로 데이터 다운로드
        df = fdr.DataReader(code, start_date)
        
        if df.empty:
            print("데이터 없음 (Skip)")
            continue
            
        df = df.reset_index()

        # [수정 1] Date 컬럼 시간 제거
        df['Date'] = df['Date'].dt.date
        
        # [수정 2] Change 컬럼: 비율 -> % 변환 및 반올림
        if 'Change' in df.columns:
            # 0.0152 -> 1.52로 변환
            df['Change'] = (df['Change'] * 100).round(2)

        df['Code'] = code
        df['Name'] = name
        
        # 컬럼 선택
        cols = ['Date', 'Code', 'Name', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]

        # DB에 저장
        df.to_sql(name=table_name, 
                  con=engine, 
                  if_exists='append', 
                  index=False,
                  dtype={'Date': Date}) 
        
        print(f"완료 ({len(df)}건)")

    print("\n[SUCCESS] 모든 데이터 수집 및 저장이 완료되었습니다.")

except Exception as e:
    print(f"\n[ERROR] 작업 중 오류 발생: {e}")

finally:
    conn.close()
