"""
scheduler.py
자동매매 스케줄러
- 장 시작 전: 종목 선발
- 장 중: 매수/매도 신호 모니터링
- 장 마감 후: 결과 정리
"""

import schedule
import time
from datetime import datetime
import logging
from config import TRADING_CONFIG
from trader import ETFTrader

logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    자동매매 스케줄러
    """
    
    def __init__(self, account_password=None):
        """
        초기화
        
        Parameters:
        -----------
        account_password : str, optional
            계좌 비밀번호
        """
        self.trader = None
        self.account_password = account_password
        self.is_trading_day = True
        
        logger.info("스케줄러 초기화")
    
    def is_market_open(self):
        """
        장 운영 시간 확인
        
        Returns:
        --------
        bool
            장 운영 중 여부
        """
        now = datetime.now()
        
        # 주말 체크
        if now.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
        
        # 시간 체크
        current_time = now.strftime('%H:%M:%S')
        market_open = TRADING_CONFIG['market_open']
        market_close = TRADING_CONFIG['market_close']
        
        return market_open <= current_time <= market_close
    
    def morning_routine(self):
        """
        장 시작 전 루틴 (08:50)
        - 트레이더 초기화
        - 계좌 동기화
        """
        logger.info("\n" + "=" * 60)
        logger.info("장 시작 전 루틴")
        logger.info("=" * 60)
        
        try:
            # 트레이더 생성
            if self.trader is None:
                self.trader = ETFTrader(account_password=self.account_password)
            
            # 계좌 동기화
            self.trader.sync_positions_with_account()
            
            # 계좌 정보 출력
            account_info = self.trader.get_account_info()
            
            logger.info("장 시작 전 루틴 완료")
        
        except Exception as e:
            logger.error(f"장 시작 전 루틴 오류: {e}")
    
    def trading_routine(self):
        """
        장 중 루틴 (매 1분마다)
        - 매도 신호 확인
        - 매수 신호 확인 (리밸런싱 필요 시)
        """
        if not self.is_market_open():
            return
        
        try:
            if self.trader is None:
                logger.warning("트레이더 미초기화")
                return
            
            # 매도 신호 확인
            self.trader.check_sell_signals()
            
            # 리밸런싱 필요 시
            if self.trader.need_rebalancing():
                self.trader.rebalance()
        
        except Exception as e:
            logger.error(f"장 중 루틴 오류: {e}")
    
    def evening_routine(self):
        """
        장 마감 후 루틴 (15:30)
        - 일일 결과 정리
        - 상태 저장
        """
        logger.info("\n" + "=" * 60)
        logger.info("장 마감 후 루틴")
        logger.info("=" * 60)
        
        try:
            if self.trader is None:
                return
            
            # 계좌 정보 조회
            account_info = self.trader.get_account_info()
            
            # 포지션 정보 출력
            logger.info(f"보유 포지션: {len(self.trader.positions)}개")
            for code, pos in self.trader.positions.items():
                data = self.trader.db.get_latest_data(code)
                name = data.get('Name', 'Unknown') if data else 'Unknown'
                logger.info(f"  - {name} ({code}): {pos['quantity']}주, "
                          f"보유일={pos['hold_days']}")
            
            # 상태 저장
            self.trader.save_state()
            
            logger.info("장 마감 후 루틴 완료")
        
        except Exception as e:
            logger.error(f"장 마감 후 루틴 오류: {e}")
    
    def setup_schedule(self):
        """스케줄 설정"""
        # 장 시작 전 (08:50)
        schedule.every().day.at("08:50").do(self.morning_routine)
        
        # 장 중 (09:00 ~ 15:20, 매 1분)
        schedule.every(1).minutes.do(self.trading_routine)
        
        # 장 마감 후 (15:30)
        schedule.every().day.at("15:30").do(self.evening_routine)
        
        logger.info("스케줄 설정 완료")
        logger.info("  - 08:50: 장 시작 전 루틴")
        logger.info("  - 09:00~15:20: 매매 모니터링 (1분 간격)")
        logger.info("  - 15:30: 장 마감 후 루틴")
    
    def run(self):
        """스케줄러 실행"""
        self.setup_schedule()
        
        logger.info("\n" + "=" * 60)
        logger.info("자동매매 스케줄러 시작")
        logger.info("=" * 60 + "\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("\n스케줄러 종료 (사용자 중단)")
        
        except Exception as e:
            logger.error(f"스케줄러 오류: {e}")
        
        finally:
            if self.trader:
                self.trader.close()


# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_scheduler.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    import getpass
    
    # 비밀번호 입력
    password = getpass.getpass("계좌 비밀번호 (4자리, 엔터=빈값): ")
    if not password:
        password = ""
    
    # 스케줄러

    
    # 스케줄러  실행
    scheduler = TradingScheduler(account_password=password)
    scheduler.run()

