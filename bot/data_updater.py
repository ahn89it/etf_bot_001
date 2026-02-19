"""
data_updater.py
ETF 데이터 자동 업데이트
"""

import FinanceDataReader as fdr
import pymysql
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import DB_CONFIG, ETF_UNIVERSE

logger = logging.getLogger(__name__)


class DataUpdater:
    """ETF 데이터 자동 업데이트"""
    
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
                charset='utf8'
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
    
    def update_price_data(self, days_back=30):
        """
        가격 데이터 업데이트
        
        Parameters:
        -----------
        days_back : int
            과거 며칠치 데이터 수집
        """
        logger.info(f"가격 데이터 업데이트 시작 (최근 {days_back}일)")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        cursor = self.connection.cursor()
        
        success_count = 0
        fail_count = 0
        
        for code, info in ETF_UNIVERSE.items():
            name = info['name']
            
            try:
                logger.info(f"수집 중: {name} ({code})")
                
                # FinanceDataReader로 데이터 수집
                df = fdr.DataReader(code, start_date, end_date)
                
                if df.empty:
                    logger.warning(f"  데이터 없음: {code}")
                    fail_count += 1
                    continue
                
                # DB 저장
                insert_count = 0
                for date, row in df.iterrows():
                    query = """
                    INSERT INTO kr_etf (Code, Name, Date, Open, High, Low, Close, Volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        Open = VALUES(Open),
                        High = VALUES(High),
                        Low = VALUES(Low),
                        Close = VALUES(Close),
                        Volume = VALUES(Volume)
                    """
                    
                    cursor.execute(query, (
                        code,
                        name,
                        date.strftime('%Y-%m-%d'),
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume'])
                    ))
                    insert_count += 1
                
                self.connection.commit()
                logger.info(f"  완료: {insert_count}일")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  오류: {code} - {e}")
                fail_count += 1
                continue
        
        cursor.close()
        
        logger.info(f"가격 데이터 업데이트 완료: 성공={success_count}, 실패={fail_count}")
        
        return success_count, fail_count
    
    def calculate_indicators(self):
        """기술적 지표 계산"""
        logger.info("지표 계산 시작")
        
        cursor = self.connection.cursor()
        
        success_count = 0
        fail_count = 0
        
        for code, info in ETF_UNIVERSE.items():
            name = info['name']
            
            try:
                logger.info(f"계산 중: {name} ({code})")
                
                # 데이터 조회
                query = f"""
                SELECT Date, Close
                FROM kr_etf
                WHERE Code = '{code}'
                ORDER BY Date
                """
                df = pd.read_sql(query, self.connection)
                
                if len(df) < 130:
                    logger.warning(f"  데이터 부족: {code} ({len(df)}일)")
                    fail_count += 1
                    continue
                
                # 지표 계산
                df['sma_20'] = df['Close'].rolling(window=20).mean()
                df['sma_60'] = df['Close'].rolling(window=60).mean()
                df['r_12w'] = df['Close'].pct_change(60) * 100   # 퍼센트
                df['r_26w'] = df['Close'].pct_change(130) * 100  # 퍼센트
                
                # DB 업데이트
                update_count = 0
                for idx, row in df.iterrows():
                    if pd.notna(row['sma_20']):
                        update_query = f"""
                        UPDATE kr_etf
                        SET sma_20 = {row['sma_20']},
                            sma_60 = {row['sma_60']},
                            r_12w = {row['r_12w'] if pd.notna(row['r_12w']) else 'NULL'},
                            r_26w = {row['r_26w'] if pd.notna(row['r_26w']) else 'NULL'}
                        WHERE Code = '{code}' AND Date = '{row['Date']}'
                        """
                        cursor.execute(update_query)
                        update_count += 1
                
                self.connection.commit()
                logger.info(f"  완료: {update_count}건")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  오류: {code} - {e}")
                fail_count += 1
                continue
        
        cursor.close()
        
        logger.info(f"지표 계산 완료: 성공={success_count}, 실패={fail_count}")
        
        return success_count, fail_count
    
    def update_all(self, days_back=30):
        """
        전체 업데이트 (가격 + 지표)
        
        Parameters:
        -----------
        days_back : int
            과거 며칠치 데이터 수집
        
        Returns:
        --------
        bool
            성공 여부
        """
        logger.info("=" * 60)
        logger.info("ETF 데이터 전체 업데이트 시작")
        logger.info("=" * 60)
        
        try:
            # 1. 가격 데이터 수집
            price_success, price_fail = self.update_price_data(days_back)
            
            # 2. 지표 계산
            indicator_success, indicator_fail = self.calculate_indicators()
            
            logger.info("=" * 60)
            logger.info("ETF 데이터 전체 업데이트 완료")
            logger.info(f"  가격 데이터: 성공={price_success}, 실패={price_fail}")
            logger.info(f"  지표 계산: 성공={indicator_success}, 실패={indicator_fail}")
            logger.info("=" * 60)
            
            return price_fail == 0 and indicator_fail == 0
            
        except Exception as e:
            logger.error(f"업데이트 오류: {e}")
            return False


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    updater = DataUpdater()
    
    # 최근 30일 데이터 업데이트
    success = updater.update_all(days_back=30)
    
    if success:
        print("\n✅ 업데이트 성공!")
    else:
        print("\n❌ 업데이트 실패!")
    
    updater.disconnect()
