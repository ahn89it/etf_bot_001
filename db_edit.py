import pandas as pd
import numpy as np
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

db_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_schema}?charset=utf8mb4"
engine = create_engine(db_url)
print(f"[INFO] DB에 접속하여 데이터를 불러옵니다...")

# 2. 데이터 불러오기
# ----------------------------------------
query = f"SELECT * FROM {table_name}"
df_all = pd.read_sql(query, engine)

# 날짜 기준 정렬
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all = df_all.sort_values(['Code', 'Date']).reset_index(drop=True)

print(f"[INFO] 총 {len(df_all)}건의 데이터를 로드했습니다. 지표 계산을 시작합니다.")

# 3. 지표 계산 함수 정의
# ----------------------------------------
def calculate_metrics(df):
    # (1) p_12w: 60거래일 전 종가
    df['p_12w'] = df['Close'].shift(60)
    
    # (2) p_26w: 130거래일 전 종가
    df['p_26w'] = df['Close'].shift(130)
    
    # (3) r_12w: 12주 수익률 (%) -> (종가 / p_12w - 1) * 100
    df['r_12w'] = ((df['Close'] / df['p_12w']) - 1) * 100
    
    # (4) r_26w: 26주 수익률 (%) -> (종가 / p_26w - 1) * 100
    df['r_26w'] = ((df['Close'] / df['p_26w']) - 1) * 100
    
    # (5) sma_20: 종가 20일 이동평균
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    # (6) sma_60: 종가 60일 이동평균
    df['sma_60'] = df['Close'].rolling(window=60).mean()
    
    # (7) vma_20: 거래량 20일 이동평균
    df['vma_20'] = df['Volume'].rolling(window=20).mean()
    
    # (8) vma_60: 거래량 60일 이동평균
    df['vma_60'] = df['Volume'].rolling(window=60).mean()
    
    # (9) atr_14: 14일 평균 변동폭
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    return df

# 4. 종목별 그룹화 및 계산 적용
# ----------------------------------------
df_calculated = df_all.groupby('Code', group_keys=False).apply(calculate_metrics)

# 5. 데이터 포맷팅 (소수점 둘째 자리 반올림)
# ----------------------------------------
target_cols = ['r_12w', 'r_26w', 'sma_20', 'sma_60', 'vma_20', 'vma_60', 'atr_14']

for col in target_cols:
    # 소수점 둘째 자리까지 반올림
    df_calculated[col] = df_calculated[col].round(2)

print("[INFO] 계산 완료. DB 업데이트 중...")

# 6. DB 저장 (덮어쓰기)
# ----------------------------------------
try:
    df_calculated.to_sql(name=table_name, 
                         con=engine, 
                         if_exists='replace', 
                         index=False,
                         dtype={'Date': Date},
                         chunksize=1000)
    
    print("[SUCCESS] 모든 지표가 % 단위 및 소수점 2자리로 업데이트되었습니다.")

except Exception as e:
    print(f"[ERROR] DB 저장 중 오류 발생: {e}")
