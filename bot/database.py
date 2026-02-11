"""
database.py
MySQL 데이터베이스 연결 및 ETF 데이터 조회
"""

import pymysql
import pandas as pd
from datetime import datetime, timedelta
from config import DB_CONFIG, ETF_UNIVERSE
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    데이터베이스 관리 클래스
    - DB 연결 관리
    - ETF 데이터 조회
    - 지표 데이터 조회
    """
    
    def __init__(self):
        """초기화"""
        self.connection = None
        self.connect()
    
    def connect(self):
        """DB 연결"""
        try:
            self.connection = pymysql.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                port=DB_CONFIG['port'],
                charset=DB_CONFIG['charset']
            )
            logger.info("DB 연결 성공")
        except Exception as e:
            logger.error(f"DB 연결 실패: {e}")
            raise
    
    def disconnect(self):
        """DB 연결 종료"""
        if self.connection:
            self.connection.close()
            logger.info("DB 연결 종료")
    
    def reconnect(self):
        """DB 재연결"""
        self.disconnect()
        self.connect()
    
    def get_etf_data(self, code, date=None, days_back=1):
        """
        특정 ETF의 데이터 조회
        
        Parameters:
        -----------
        code : str
            종목코드 (예: '091160')
        date : str, optional
            조회 기준일 (YYYY-MM-DD), None이면 오늘
        days_back : int
            조회할 과거 일수
        
        Returns:
        --------
        DataFrame
            ETF 데이터 (날짜 내림차순)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            Date, Code, Name, 
            Open, High, Low, Close, Volume,
            r_12w, r_26w, 
            sma_20, sma_60
        FROM kr_etf
        WHERE Code = '{code}' 
          AND Date <= '{date}'
        ORDER BY Date DESC
        LIMIT {days_back}
        """
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            logger.error(f"데이터 조회 실패 ({code}): {e}")
            return pd.DataFrame()
    
    def get_latest_data(self, code):
        """
        최신 데이터 조회 (오늘 또는 가장 최근)
        
        Parameters:
        -----------
        code : str
            종목코드
        
        Returns:
        --------
        dict or None
            최신 데이터
        """
        df = self.get_etf_data(code, days_back=1)
        
        if df.empty:
            return None
        
        return df.iloc[0].to_dict()
    
    def get_available_universe(self, date=None):
        """
        특정 날짜에 거래 가능한 ETF 리스트
        
        Parameters:
        -----------
        date : str, optional
            기준일 (YYYY-MM-DD)
        
        Returns:
        --------
        list
            거래 가능한 종목코드 리스트
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        date_dt = pd.to_datetime(date)
        available = []
        
        for code, info in ETF_UNIVERSE.items():
            start_dt = pd.to_datetime(info['start'])
            
            # 상장일 + 210일(약 150거래일) 이후부터 사용
            if date_dt >= start_dt + pd.Timedelta(days=210):
                available.append(code)
        
        logger.info(f"{date} 기준 사용 가능 종목: {len(available)}개")
        return available
    
    def check_data_availability(self, code, date=None):
        """
        데이터 사용 가능 여부 확인
        
        Parameters:
        -----------
        code : str
            종목코드
        date : str, optional
            기준일
        
        Returns:
        --------
        bool
            사용 가능 여부
        """
        df = self.get_etf_data(code, date, days_back=1)
        
        if df.empty:
            return False
        
        row = df.iloc[0]
        
        # 필수 데이터 확인
        required_fields = ['r_12w', 'r_26w', 'sma_20', 'sma_60', 'Close']
        
        for field in required_fields:
            if pd.isna(row[field]):
                logger.warning(f"{code}: {field} 데이터 없음")
                return False
        
        return True
    
    def get_yesterday_data(self, code):
        """
        어제 데이터 조회 (trigger 계산용)
        
        Parameters:
        -----------
        code : str
            종목코드
        
        Returns:
        --------
        dict or None
            어제 데이터
        """
        df = self.get_etf_data(code, days_back=2)
        
        if len(df) < 2:
            return None
        
        # 두 번째 행이 어제 데이터
        return df.iloc[1].to_dict()


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # DB 매니저 생성
    db = DatabaseManager()
    
    # 테스트 1: 사용 가능한 유니버스 조회
    print("\n=== 테스트 1: 사용 가능한 유니버스 ===")
    available = db.get_available_universe()
    print(f"사용 가능 종목 수: {len(available)}")
    print(f"종목 리스트: {available}")
    
    # 테스트 2: 특정 종목 데이터 조회
    print("\n=== 테스트 2: KODEX 반도체 데이터 조회 ===")
    data = db.get_latest_data('091160')
    if data:
        print(f"날짜: {data['Date']}")
        print(f"종가: {data['Close']:,}원")
        print(f"12주 수익률: {data['r_12w']:.2%}")
        print(f"26주 수익률: {data['r_26w']:.2%}")
    
    # 테스트 3: 데이터 사용 가능 여부
    print("\n=== 테스트 3: 데이터 사용 가능 여부 ===")
    for code in available[:3]:
        is_available = db.check_data_availability(code)
        name = ETF_UNIVERSE[code]['name']
        print(f"{name} ({code}): {'사용 가능' if is_available else '사용 불가'}")
    
    # 연결 종료
    db.disconnect()
