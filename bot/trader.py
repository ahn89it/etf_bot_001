"""
trader.py
메인 트레이딩 로직
- 포지션 관리
- 매수/매도 실행
- 상태 모니터링
"""

import json
import os
from datetime import datetime
import time
import logging
from config import ETF_UNIVERSE, TRADING_CONFIG
from database import DatabaseManager
from strategy import ETFStrategy
from kiwoom_api import KiwoomAPI

logger = logging.getLogger(__name__)


class ETFTrader:
    """ETF 자동매매 트레이더"""

    
    def __init__(self, account_password=None):
        """초기화"""
        logger.info("=" * 60)
        logger.info("트레이더 초기화 시작")
        logger.info("=" * 60)
        
        # 1. DB 연결
        logger.info("1/5 DB 연결 중...")
        self.db = DatabaseManager()
        logger.info("    ✓ DB 연결 완료")
        
        # 2. 전략 초기화
        logger.info("2/5 전략 초기화 중...")
        self.strategy = ETFStrategy(self.db)
        logger.info("    ✓ 전략 초기화 완료")
        
        # 3. 키움 API 초기화
        logger.info("3/5 키움 API 초기화 중...")
        self.api = KiwoomAPI(account_password)
        logger.info("    ✓ 키움 API 초기화 완료")
        
        # 4. 로그인
        logger.info("4/5 로그인 중...")
        if not self.api.login():
            raise Exception("키움 API 로그인 실패")
        logger.info("    ✓ 로그인 완료")
        
        # 5. 상태 로드
        logger.info("5/5 상태 로드 중...")
        self.positions = {}
        self.selected_stocks = []
        self.pending_orders = set()  # ★★★ 추가: 주문 중인 종목 ★★★
        self.state_file = './trading_state.json'
        self.load_state()
        logger.info("    ✓ 상태 로드 완료")
        
        logger.info("=" * 60)
        logger.info("트레이더 초기화 완료!")
        logger.info("=" * 60)

    
    def save_state(self):
        """상태 저장"""
        state = {
            'positions': self.positions,
            'selected_stocks': self.selected_stocks,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug("상태 저장 완료")
        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")
    
    def load_state(self):
        """상태 로드"""
        if not os.path.exists(self.state_file):
            logger.info("    저장된 상태 없음 (신규 시작)")
            return
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.positions = state.get('positions', {})
            self.selected_stocks = state.get('selected_stocks', [])
            
            logger.info(f"    포지션 {len(self.positions)}개 로드")
        except Exception as e:
            logger.error(f"상태 로드 실패: {e}")
    
    def get_account_info(self):
        """계좌 정보 조회"""
        logger.info("\n" + "=" * 60)
        logger.info("계좌 정보 조회")
        logger.info("=" * 60)
        
        deposit = self.api.get_deposit()
        balance = self.api.get_balance()
        
        logger.info(f"예수금: {deposit:,}원")
        logger.info(f"보유 종목: {len(balance)}개")
        
        for code, info in balance.items():
            logger.info(f"  - {info['name']} ({code}): {info['quantity']}주 @ {info['price']:,}원")
        
        return {'deposit': deposit, 'balance': balance}
    
    def sync_positions_with_account(self):
        """계좌 잔고와 포지션 동기화"""
        logger.info("포지션 동기화 중...")
        
        balance = self.api.get_balance()
        
        # 계좌에 있는데 포지션에 없는 경우
        for code, info in balance.items():
            if code not in self.positions:
                logger.warning(f"{code} 계좌에 있으나 포지션 없음 → 추가")
                self.positions[code] = {
                    'entry_date': datetime.now().strftime('%Y-%m-%d'),
                    'entry_price': info['buy_price'],
                    'quantity': info['quantity'],
                    'hold_days': 0
                }
        
        # 포지션에 있는데 계좌에 없는 경우
        codes_to_remove = []
        for code in self.positions:
            if code not in balance:
                logger.warning(f"{code} 포지션에 있으나 계좌 없음 → 제거")
                codes_to_remove.append(code)
        
        for code in codes_to_remove:
            del self.positions[code]
        
        self.save_state()
        logger.info("포지션 동기화 완료")
    
    def select_stocks(self):
        """종목 선발"""
        logger.info("\n" + "=" * 60)
        logger.info("종목 선발 시작")
        logger.info("=" * 60)
        
        self.selected_stocks = self.strategy.calculate_momentum_score()
        
        if not self.selected_stocks:
            logger.warning("선발된 종목 없음")
            return []
        
        logger.info(f"\n선발 완료: {len(self.selected_stocks)}개 종목")
        self.save_state()
        
        return self.selected_stocks
    
    def check_buy_signals(self):
        """매수 신호 확인 및 실행"""
        if not self.selected_stocks:
            logger.info("선발된 종목 없음")
            return []
        
        logger.info("\n" + "=" * 60)
        logger.info("매수 신호 확인")
        logger.info("=" * 60)
        
        buy_candidates = []
        
        for code, score, name in self.selected_stocks:
            # ★★★ 이미 보유 중인 종목 제외 ★★★
            if code in self.positions:
                logger.info(f"{name} ({code}): 이미 보유 중")
                continue
            
            # ★★★ 주문 처리 중인 종목 제외 (중복 주문 방지) ★★★
            if code in self.pending_orders:
                logger.info(f"{name} ({code}): 주문 처리 중")
                continue
            
            try:
                # 실시간 현재가 조회
                logger.info(f"{name} ({code}): 현재가 조회 중...")
                current_price = self.api.get_current_price(code)
                
                if current_price == 0:
                    logger.warning(f"{name} ({code}): 현재가 조회 실패")
                    continue
                
                logger.info(f"{name} ({code}): 실시간 현재가 {current_price:,.0f}원")
                
                # 매수 신호 확인 (현재가 전달)
                signal = self.strategy.check_buy_signal(code, current_price)
                
                if signal and signal['signal']:
                    buy_candidates.append({
                        'code': code,
                        'name': name,
                        'trigger_price': signal['trigger_price']
                    })
                    logger.info(f"{name} ({code}): 매수 신호 발생!")
                else:
                    logger.info(f"{name} ({code}): 매수 조건 미충족")
                
            except Exception as e:
                logger.error(f"{name} ({code}) 매수 신호 확인 오류: {e}")
                continue
        
        if not buy_candidates:
            logger.info("매수 신호 없음")
            return []
        
        return self.execute_buy_orders(buy_candidates)


    
    def execute_buy_orders(self, buy_candidates):
        """매수 주문 실행"""
        logger.info("\n" + "=" * 60)
        logger.info(f"매수 실행: {len(buy_candidates)}개 종목")
        logger.info("=" * 60)
        
        # 예수금 조회
        total_deposit = self.api.get_deposit()
        logger.info(f"전체 예수금: {total_deposit:,}원")
        
        # 투자 금액 계산 (기존 코드 유지)
        try:
            from config import INVESTMENT_CONFIG
            
            if INVESTMENT_CONFIG.get('use_fixed', False):
                available_cash = INVESTMENT_CONFIG['fixed_amount']
                logger.info(f"고정 금액 모드: {available_cash:,}원")
            elif 'per_stock_amount' in INVESTMENT_CONFIG:
                per_stock = INVESTMENT_CONFIG['per_stock_amount']
                available_cash = per_stock * len(buy_candidates)
                logger.info(f"종목당 금액 모드: {per_stock:,}원 × {len(buy_candidates)}개")
            else:
                use_ratio = INVESTMENT_CONFIG.get('use_ratio', 1.0)
                available_cash = int(total_deposit * use_ratio)
                logger.info(f"비율 모드: {use_ratio*100}% = {available_cash:,}원")
            
            min_reserve = INVESTMENT_CONFIG.get('min_cash_reserve', 0)
            if min_reserve > 0:
                if total_deposit - available_cash < min_reserve:
                    available_cash = max(0, total_deposit - min_reserve)
                    logger.info(f"최소 현금 보유: {min_reserve:,}원 유지")
            
            max_investment = INVESTMENT_CONFIG.get('max_investment', float('inf'))
            if available_cash > max_investment:
                available_cash = max_investment
                logger.info(f"최대 투자 제한: {max_investment:,}원")
            
            if available_cash > total_deposit:
                available_cash = total_deposit
                logger.warning(f"예수금 부족: {total_deposit:,}원만 사용")
            
        except ImportError:
            available_cash = total_deposit
            logger.info("투자 설정 없음 - 전액 사용")
        
        logger.info(f"실제 투자 금액: {available_cash:,}원")
        
        if available_cash < 10000:
            logger.error("투자 가능 금액 부족 (최소 1만원)")
            return []
        
        # 균등 분할
        num_stocks = len(buy_candidates)
        allocation_per_stock = available_cash / num_stocks
        logger.info(f"종목당 배분: {allocation_per_stock:,.0f}원")
        
        bought_stocks = []
        
        for candidate in buy_candidates:
            code = candidate['code']
            name = candidate['name']
            trigger_price = candidate['trigger_price']
            
            try:
                # ★★★ 주문 중 플래그 설정 (중복 주문 방지) ★★★
                self.pending_orders.add(code)
                
                # 수량 계산
                quantity = self.strategy.calculate_position_size(
                    allocation_per_stock, 1, trigger_price
                )
                
                if quantity == 0:
                    logger.warning(f"{name}: 수량 0 (매수 불가)")
                    self.pending_orders.discard(code)
                    continue
                
                # 매수 주문
                logger.info(f"{name} 매수 주문: {quantity}주 @ {trigger_price:,.0f}원")
                
                result = self.api.buy(code, quantity, int(trigger_price))
                
                # 체결 확인
                if not result:
                    logger.error(f"{name} 주문 결과 없음")
                    self.pending_orders.discard(code)
                    continue
                
                if result.get('success') and result.get('status') == '체결':
                    # 실제 체결 완료
                    exec_qty = result.get('exec_qty', quantity)
                    exec_price = result.get('exec_price', trigger_price)
                    
                    # ★★★ 중복 체결 처리 (수량 합산) ★★★
                    if code in self.positions:
                        logger.warning(f"{name} 중복 체결 감지! 수량 합산")
                        
                        existing = self.positions[code]
                        total_qty = existing['quantity'] + exec_qty
                        
                        # 평균 단가 계산
                        total_cost = (existing['entry_price'] * existing['quantity'] + 
                                    exec_price * exec_qty)
                        avg_price = total_cost / total_qty
                        
                        self.positions[code]['quantity'] = total_qty
                        self.positions[code]['entry_price'] = int(avg_price)
                        
                        logger.info(f"  기존: {existing['quantity']}주 @ {existing['entry_price']:,}원")
                        logger.info(f"  추가: {exec_qty}주 @ {exec_price:,}원")
                        logger.info(f"  합계: {total_qty}주 @ {int(avg_price):,}원")
                    else:
                        # 신규 포지션
                        self.positions[code] = {
                            'entry_date': datetime.now().strftime('%Y-%m-%d'),
                            'entry_time': datetime.now().strftime('%H:%M:%S'),
                            'entry_price': exec_price,
                            'quantity': exec_qty,
                            'order_no': result.get('order_no', '')
                        }
                        logger.info(f"✅ {name} 매수 체결 완료: {exec_qty}주 @ {exec_price:,}원")
                    
                    bought_stocks.append(code)
                    
                    # ★★★ 주문 중 플래그 해제 ★★★
                    self.pending_orders.discard(code)
                    
                    time.sleep(TRADING_CONFIG['order_delay'])
                
                elif result.get('status') == '접수':
                    logger.warning(f"⚠️ {name} 주문 접수됨 (미체결) - 포지션 저장 안 함")
                    # ★★★ 플래그는 유지 (다음 루틴에서 중복 주문 방지) ★★★
                
                else:
                    logger.error(f"❌ {name} 매수 실패: {result}")
                    self.pending_orders.discard(code)
            
            except Exception as e:
                logger.error(f"{name} 매수 오류: {e}")
                self.pending_orders.discard(code)
                continue
        
        # 상태 저장
        self.save_state()
        
        logger.info(f"\n매수 완료: {len(bought_stocks)}개 종목")
        return bought_stocks


    def check_sell_signals(self):
        """매도 신호 확인 및 실행"""
        if not self.positions:
            logger.info("보유 포지션 없음")
            return []
        
        logger.info("\n" + "=" * 60)
        logger.info("매도 신호 확인")
        logger.info("=" * 60)
        
        sell_candidates = []
        
        for code, position in list(self.positions.items()):
            try:
                name = ETF_UNIVERSE.get(code, {}).get('name', 'Unknown')
                
                # ★★★ 영업일 기준 보유일 계산 ★★★
                from utils import get_business_days
                
                entry_date = position['entry_date']  # 'YYYY-MM-DD' 문자열
                today = datetime.now().strftime('%Y-%m-%d')
                
                hold_days = get_business_days(entry_date, today)
                
                logger.info(f"{name} ({code}): "
                        f"보유일={hold_days}일 (영업일), "
                        f"진입일={position['entry_date']}, "
                        f"진입가={position['entry_price']:,.0f}원")
                
                # 매도 신호 확인 (hold_days 전달)
                signal = self.strategy.check_sell_signal(code, position, hold_days)
                
                if signal and signal['signal']:
                    sell_candidates.append({
                        'code': code,
                        'name': name,
                        'position': position,
                        'reason': signal['reason'],
                        'sell_price': signal.get('sell_price', 0)
                    })
                    logger.info(f"🔴 {name}: 매도 신호 ({signal['reason']})")
                else:
                    logger.info(f"  → 보유 유지")
            
            except Exception as e:
                logger.error(f"{code} 매도 신호 확인 오류: {e}")
                continue
        
        if not sell_candidates:
            logger.info("매도 신호 없음")
            return []
        
        return self.execute_sell_orders(sell_candidates)

    
    def execute_sell_orders(self, sell_candidates):
        """매도 주문 실행"""
        logger.info("\n" + "=" * 60)
        logger.info(f"매도 실행: {len(sell_candidates)}개 종목")
        logger.info("=" * 60)
        
        sold_stocks = []
        
        for candidate in sell_candidates:
            code = candidate['code']
            name = candidate['name']
            position = candidate['position']
            reason = candidate['reason']
            
            try:
                quantity = position['quantity']
                
                logger.info(f"{name} 매도 주문: {quantity}주 (사유: {reason})")
                
                # 시장가 매도
                result = self.api.sell(code, quantity, 0)
                
                # ★★★ 체결 확인 ★★★
                if result.get('success') and result.get('status') == '체결':
                    entry_price = position['entry_price']
                    sell_price = result.get('exec_price', candidate.get('sell_price', 0))
                    
                    if sell_price > 0:
                        profit_pct = (sell_price / entry_price - 1) * 100
                    else:
                        profit_pct = 0
                    
                    logger.info(f"✅ {name} 매도 성공!")
                    logger.info(f"  진입가: {entry_price:,.0f}원")
                    logger.info(f"  매도가: {sell_price:,.0f}원")
                    logger.info(f"  수익률: {profit_pct:+.2f}%")
                    
                    del self.positions[code]
                    sold_stocks.append(code)
                    
                    time.sleep(TRADING_CONFIG['order_delay'])
                
                elif result.get('status') == '접수':
                    logger.warning(f"⚠️ {name} 매도 주문 접수됨 (미체결)")
                
                else:
                    logger.error(f"❌ {name} 매도 실패: {result}")
            
            except Exception as e:
                logger.error(f"{name} 매도 오류: {e}")
                continue
        
        self.save_state()
        logger.info(f"\n매도 완료: {len(sold_stocks)}개 종목")
        return sold_stocks

    def need_rebalancing(self):
        """리밸런싱 필요 여부"""
        return len(self.positions) == 0
    
    def rebalance(self):
        """리밸런싱 실행"""
        logger.info("\n" + "=" * 60)
        logger.info("리밸런싱 시작")
        logger.info("=" * 60)
        
        self.select_stocks()
        bought = self.check_buy_signals()
        
        if bought:
            logger.info(f"리밸런싱 완료: {len(bought)}개 종목 매수")
        else:
            logger.info("리밸런싱 완료: 매수 없음")
    
    def daily_routine(self):
        """일일 거래 루틴"""
        logger.info("\n" + "=" * 60)
        logger.info(f"일일 루틴 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            self.sync_positions_with_account()
            account_info = self.get_account_info()
            sold = self.check_sell_signals()
            
            if self.need_rebalancing():
                logger.info("리밸런싱 필요")
                self.rebalance()
            else:
                logger.info(f"리밸런싱 불필요 (포지션 {len(self.positions)}개)")
            
            logger.info("=" * 60)
            logger.info("일일 루틴 완료")
            logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"일일 루틴 오류: {e}")
            import  traceback
            traceback.print_exc()
    
    def close(self):
        """종료"""
        self.save_state()
        self.db.disconnect()
        logger.info("트레이더 종료")


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == "__main__":
    import getpass
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trader_test.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    try:
        password = getpass.getpass("계좌 비밀번호 (4자리, 엔터=빈값): ")
        if not password:
            password = ""
        
        trader = ETFTrader(account_password=password)
        trader.daily_routine()
        trader.close()
    
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

