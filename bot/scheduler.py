"""
scheduler.py
자동매매 스케줄러
"""

import schedule
import time
from datetime import datetime
import logging
from config import TRADING_CONFIG, ETF_UNIVERSE
from trader import ETFTrader
from data_updater import DataUpdater

logger = logging.getLogger(__name__)


class TradingScheduler:
    """자동매매 스케줄러"""
    
    def __init__(self, account_password=None):
        """초기화"""
        self.trader = None
        self.account_password = account_password
        self.updater = DataUpdater()
        self.is_trading_day = True
        
        logger.info("스케줄러 초기화")
    
    def is_market_open(self):
        """장 운영 시간 확인"""
        now = datetime.now()
        
        # 주말 체크
        if now.weekday() >= 5:
            return False
        
        # 시간 체크
        current_time = now.strftime('%H:%M:%S')
        market_open = TRADING_CONFIG['market_open']
        market_close = TRADING_CONFIG['market_close']
        
        return market_open <= current_time <= market_close
    
    def data_update_routine(self):
        """데이터 업데이트 루틴 (08:00)"""
        
        # 오늘 이미 업데이트했으면 스킵
        today = datetime.now().strftime('%Y-%m-%d')
        if hasattr(self, 'last_update_date') and self.last_update_date == today:
            logger.info("오늘 이미 데이터 업데이트 완료 (스킵)")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("데이터 업데이트 루틴 시작")
        logger.info(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            # 최근 5일 데이터 업데이트
            success = self.updater.update_all(days_back=5)
            
            if success:
                logger.info("✅ 데이터 업데이트 성공")
                self.last_update_date = today
            else:
                logger.warning("⚠️ 데이터 업데이트 일부 실패")
            
        except Exception as e:
            logger.error(f"데이터 업데이트 오류: {e}")
    
    def check_and_update_if_needed(self):
        """시작 시 데이터 업데이트 필요 여부 확인"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        
        # 08:00 이후에 시작했으면 즉시 업데이트
        if current_time >= "08:00":
            logger.info(f"현재 시간 {current_time} → 즉시 데이터 업데이트 실행")
            self.data_update_routine()
    
    def morning_routine(self):
        """장 시작 전 루틴 (08:50)"""
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
        """장 중 루틴 (매 1분마다)"""
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
        """장 마감 후 루틴 (15:30)"""
        logger.info("\n" + "=" * 60)
        logger.info("장 마감 후 루틴")
        logger.info("=" * 60)
        
        try:
            if self.trader is None:
                logger.warning("트레이더 미초기화")
                return
            
            # 계좌 정보 조회
            account_info = self.trader.get_account_info()
            
            # 포지션 정보 출력
            logger.info(f"보유 포지션: {len(self.trader.positions)}개")
            
            for code, position in self.trader.positions.items():
                # ETF 이름 조회
                name = ETF_UNIVERSE.get(code, {}).get('name', 'Unknown')
                
                # ★★★ 영업일 기준 보유일 계산 ★★★
                from utils import get_business_days
                
                entry_date = position['entry_date']
                today = datetime.now().strftime('%Y-%m-%d')
                
                hold_days = get_business_days(entry_date, today)
                
                logger.info(f"  {name} ({code}): "
                        f"{position['quantity']}주, "
                        f"진입가={position['entry_price']:,}원, "
                        f"보유일={hold_days}일 (영업일)")
            
            # 주문 중 플래그 초기화
            if hasattr(self.trader, 'pending_orders'):
                self.trader.pending_orders.clear()
                logger.info("주문 중 플래그 초기화 완료")
            
            # 상태 저장
            self.trader.save_state()
            
            logger.info("장 마감 후 루틴 완료")
        
        except Exception as e:
            logger.error(f"장 마감 후 루틴 오류: {e}")
            import traceback
            traceback.print_exc()

    
    def setup_schedule(self):
        """스케줄 설정"""
        
        # 데이터 업데이트 (08:00)
        schedule.every().day.at("08:00:00").do(self.data_update_routine)
        
        # 장 시작 전 (08:50)
        schedule.every().day.at("08:50:00").do(self.morning_routine)
        
        # 장 중 (매 1분)
        schedule.every(1).minutes.do(self.trading_routine)
        
        # 장 마감 후 (15:30)
        schedule.every().day.at("15:30:00").do(self.evening_routine)
        
        logger.info("=" * 60)
        logger.info("스케줄 설정 완료")
        logger.info("=" * 60)
        logger.info("  - 08:00:00: 데이터 자동 업데이트")
        logger.info("  - 08:50:00: 장 시작 전 루틴")
        logger.info("  - 09:00~15:20: 매매 모니터링 (1분 간격)")
        logger.info("  - 15:30:00: 장 마감 후 루틴")
        logger.info("=" * 60)
        
        # 등록된 스케줄 확인
        logger.info("\n등록된 스케줄:")
        for job in schedule.get_jobs():
            logger.info(f"  - {job}")
    
    def run(self):
        """스케줄러 실행"""
        
        # ★★★ 시작 시 데이터 업데이트 필요 여부 확인 ★★★
        self.check_and_update_if_needed()
        
        # 스케줄 설정
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
            if self.updater:
                self.updater.disconnect()


# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    import getpass
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_scheduler.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 비밀번호 입력
    password = getpass.getpass("계좌 비밀번호 (4자리, 엔터=빈값): ")
    if not password:
        password = ""
    
    # 스케줄러 실행
    scheduler = TradingScheduler(account_password=password)
    scheduler.run()
