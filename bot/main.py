# main.py 수정 (완전한 버전)

"""
main.py
ETF 퀀트 자동매매 시스템 메인 실행 파일
"""

import sys
import signal
import logging
from datetime import datetime
from config import LOGGING_CONFIG
from scheduler import TradingScheduler
from trader import ETFTrader
import os
import getpass


# ============================================================================
# 시그널 핸들러 (Ctrl+C 처리)
# ============================================================================
def signal_handler(sig, frame):
    """Ctrl+C 시그널 핸들러"""
    print("\n\n프로그램을 종료합니다...")
    sys.exit(0)

# 시그널 등록
signal.signal(signal.SIGINT, signal_handler)


def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = LOGGING_CONFIG['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로그 파일명 (날짜 포함)
    log_file = os.path.join(
        log_dir,
        f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['log_level']),
        format=LOGGING_CONFIG['log_format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("ETF 퀀트 자동매매 시스템 시작")
    logger.info("=" * 80)
    logger.info(f"로그 파일: {log_file}")
    
    return logger


def print_banner():
    """시작 배너 출력"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           ETF 퀀트 자동매매 시스템 v1.0                        ║
    ║                                                               ║
    ║   전략: 모멘텀 기반 ETF 자동매매                               ║
    ║   목표 수익률: 연평균 21.22%                                   ║
    ║   목표 MDD: -32.35%                                           ║
    ║                                                               ║
    ║   종료: Ctrl+C 또는 창 닫기                                    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def manual_mode():
    """수동 모드 (테스트용)"""
    logger = logging.getLogger(__name__)
    
    print("\n=== 수동 모드 ===")
    print("1. 일일 루틴 실행")
    print("2. 종목 선발만")
    print("3. 계좌 정보 조회")
    print("4. 포지션 확인")
    print("0. 종료")
    
    choice = input("\n선택: ")
    
    try:
        # 비밀번호 입력
        password = None
        if choice in ["1", "3"]:
            password = getpass.getpass("계좌 비밀번호 (4자리, 엔터=빈값): ")
            if not password:
                password = ""
        
        trader = ETFTrader(account_password=password)
        
        if choice == "1":
            # 일일 루틴
            trader.daily_routine()
        
        elif choice == "2":
            # 종목 선발
            stocks = trader.select_stocks()
            print("\n선발된 종목:")
            for code, score, name in stocks:
                print(f"  {name} ({code}): {score:.4f}")
        
        elif choice == "3":
            # 계좌 정보
            info = trader.get_account_info()
            print(f"\n예수금: {info['deposit']:,}원")
            print(f"보유 종목: {len(info['balance'])}개")
            for code, data in info['balance'].items():
                print(f"  {data['name']} ({code}): {data['quantity']}주")
        
        elif choice == "4":
            # 포지션 확인
            print(f"\n포지션: {len(trader.positions)}개")
            for code, pos in trader.positions.items():
                name = trader.db.get_latest_data(code)
                if name:
                    name = name.get('Name', 'Unknown')
                print(f"  {name} ({code}):")
                print(f"    수량: {pos['quantity']}주")
                print(f"    진입가: {pos['entry_price']:,.0f}원")
                print(f"    보유일: {pos['hold_days']}일")
        
        elif choice == "0":
            print("종료합니다.")
        
        else:
            print("잘못된 선택입니다.")
        
        trader.close()
    
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
    except Exception as e:
        logger.error(f"수동 모드 오류: {e}")
        import traceback
        traceback.print_exc()


def auto_mode():
    """자동 모드 (스케줄러)"""
    logger = logging.getLogger(__name__)
    
    try:
        # 비밀번호 입력
        password = getpass.getpass("계좌 비밀번호 (4자리, 엔터=빈값): ")
        if not password:
            password = ""
        
        scheduler = TradingScheduler(account_password=password)
        scheduler.run()
    
    except KeyboardInterrupt:
        print("\n\n스케줄러를 종료합니다.")
    except Exception as e:
        logger.error(f"자동 모드 오류: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    # 로깅 설정
    logger = setup_logging()
    
    # 배너 출력
    print_banner()
    
    # 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 자동 모드 (스케줄러)")
    print("2. 수동 모드 (테스트)")
    print("0. 종료")
    
    choice = input("\n선택: ")
    
    if choice == "1":
        logger.info("자동 모드 시작")
        auto_mode()
    
    elif choice == "2":
        logger.info("수동 모드 시작")
        manual_mode()
    
    elif choice == "0":
        logger.info("프로그램 종료")
        print("종료합니다.")
    
    else:
        print("잘못된 선택입니다.")
    
    logger.info("=" * 80)
    logger.info("프로그램 종료")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
