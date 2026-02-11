"""
strategy.py
ETF 퀀트 전략 로직
- 모멘텀 계산
- 종목 선발
- 매수/매도 신호 생성
"""

import pandas as pd
from datetime import datetime
import logging
from config import STRATEGY_PARAMS, ETF_UNIVERSE
from database import DatabaseManager

logger = logging.getLogger(__name__)


class ETFStrategy:
    """ETF 퀀트 전략 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Parameters:
        -----------
        db_manager : DatabaseManager
            데이터베이스 매니저 인스턴스
        """
        self.db = db_manager
        
        # 전략 파라미터
        self.k = STRATEGY_PARAMS['k']
        self.top_n = STRATEGY_PARAMS['top_n']
        self.hold_days = STRATEGY_PARAMS['hold_days']
        self.sma_buffer = STRATEGY_PARAMS['sma_buffer']
        self.momentum_weight_12w = STRATEGY_PARAMS['momentum_weight_12w']
        self.momentum_weight_26w = STRATEGY_PARAMS['momentum_weight_26w']
        
        logger.info(f"전략 초기화: k={self.k}, top_n={self.top_n}, hold_days={self.hold_days}")
    
    def calculate_momentum_score(self, date=None):
        """
        모멘텀 점수 계산 및 상위 N개 선발
        
        Parameters:
        -----------
        date : str, optional
            평가 기준일 (YYYY-MM-DD)
        
        Returns:
        --------
        list
            [(code, score, name), ...] 상위 N개 종목
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"모멘텀 계산 시작: {date}")
        
        # 1. 사용 가능한 유니버스 조회
        available = self.db.get_available_universe(date)
        
        if not available:
            logger.warning("사용 가능한 종목이 없습니다")
            return []
        
        # 2. 각 종목별 모멘텀 점수 계산
        scores = {}
        
        for code in available:
            try:
                # 데이터 조회
                df = self.db.get_etf_data(code, date, days_back=1)
                
                if df.empty:
                    logger.debug(f"{code}: 데이터 없음")
                    continue
                
                row = df.iloc[0]
                
                # 데이터 유효성 검증
                if pd.isna(row['r_12w']) or pd.isna(row['r_26w']):
                    logger.debug(f"{code}: 모멘텀 데이터 없음")
                    continue
                
                if pd.isna(row['Close']) or pd.isna(row['sma_60']):
                    logger.debug(f"{code}: 가격/지표 데이터 없음")
                    continue
                
                # 진입 필터: 종가 > SMA(60)
                if row['Close'] <= row['sma_60']:
                    logger.debug(f"{code}: SMA(60) 필터 탈락")
                    continue
                
                # 수익률 변환 (퍼센트 → 소수)
                r_12w = row['r_12w'] / 100
                r_26w = row['r_26w'] / 100
                
                # 모멘텀 점수 계산
                score = (self.momentum_weight_12w * r_12w + 
                        self.momentum_weight_26w * r_26w)
                
                scores[code] = {
                    'score': score,
                    'name': ETF_UNIVERSE[code]['name'],
                    'r_12w': row['r_12w'],
                    'r_26w': row['r_26w'],
                    'close': row['Close']
                }
                
                logger.debug(f"{code}: score={score:.4f}")
                
            except Exception as e:
                logger.error(f"{code} 모멘텀 계산 오류: {e}")
                continue
        
        # 3. 점수 기준 정렬 및 상위 N개 선발
        if not scores:
            logger.warning("모멘텀 점수 계산된 종목이 없습니다")
            return []
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        top_n = sorted_scores[:self.top_n]
        
        # 4. 결과 로깅
        logger.info(f"=== 모멘텀 선발 결과 (상위 {self.top_n}개) ===")
        result = []
        for i, (code, info) in enumerate(top_n, 1):
            logger.info(f"{i}. {info['name']} ({code}): "
                       f"score={info['score']:.4f}, "
                       f"r_12w={info['r_12w']:.2f}%, "
                       f"r_26w={info['r_26w']:.2f}%")
            result.append((code, info['score'], info['name']))
        
        return result
    
    def check_buy_signal(self, code, current_price=None):
        """
        매수 신호 확인
        
        Parameters:
        -----------
        code : str
            종목코드
        current_price : float, optional
            현재가
        
        Returns:
        --------
        dict or None
            매수 신호 정보
        """
        try:
            df = self.db.get_etf_data(code, days_back=2)
            
            if len(df) < 2:
                logger.warning(f"{code}: 데이터 부족")
                return None
            
            today = df.iloc[0]
            yesterday = df.iloc[1]
            
            # Trigger 가격 계산
            yesterday_range = yesterday['High'] - yesterday['Low']
            trigger_price = today['Open'] + self.k * yesterday_range
            
            # 현재가
            if current_price is None:
                current_price = today['High']
            
            # 매수 조건
            signal = current_price >= trigger_price
            
            result = {
                'signal': signal,
                'trigger_price': trigger_price,
                'current_price': current_price,
                'yesterday_range': yesterday_range,
                'today_open': today['Open'],
                'today_high': today['High']
            }
            
            if signal:
                logger.info(f"{code} 매수 신호!")
                logger.info(f"  Trigger: {trigger_price:,.0f}원")
                logger.info(f"  현재가: {current_price:,.0f}원")
            
            return result
            
        except Exception as e:
            logger.error(f"{code} 매수 신호 확인 오류: {e}")
            return None
    
    def check_sell_signal(self, code, position, current_price=None):
        """
        매도 신호 확인
        
        Parameters:
        -----------
        code : str
            종목코드
        position : dict
            포지션 정보
        current_price : float, optional
            현재가
        
        Returns:
        --------
        dict or None
            매도 신호 정보
        """
        try:
            df = self.db.get_etf_data(code, days_back=1)
            
            if df.empty:
                logger.warning(f"{code}: 데이터 없음")
                return None
            
            today = df.iloc[0]
            
            # 매수 당일 체크
            today_str = datetime.now().strftime('%Y-%m-%d')
            if position['entry_date'] == today_str:
                logger.debug(f"{code}: 매수 당일")
                return {'signal': False, 'reason': '매수 당일'}
            
            # 현재가
            if current_price is None:
                current_price = today['Close']
            
            # 조건 1: 손절
            if not pd.isna(today['sma_20']):
                sma20_threshold = today['sma_20'] * self.sma_buffer
                
                if today['Low'] < sma20_threshold:
                    logger.info(f"{code} 손절 신호!")
                    
                    return {
                        'signal': True,
                        'reason': 'SMA(20) 이탈',
                        'sell_price': current_price,
                        'sma20': today['sma_20'],
                        'low': today['Low']
                    }
            
            # 조건 2: 보유기간 도달
            if position['hold_days'] >= self.hold_days:
                logger.info(f"{code} 보유기간 도달 ({position['hold_days']}일)")
                
                return {
                    'signal': True,
                    'reason': f"{self.hold_days}일 도달",
                    'sell_price': current_price,
                    'hold_days': position['hold_days']
                }
            
            # 매도 조건 불만족
            return {'signal': False, 'reason': '보유 유지'}
            
        except Exception as e:
            logger.error(f"{code} 매도 신호 확인 오류: {e}")
            return None
    
    def calculate_position_size(self, available_cash, num_stocks, stock_price):
        """
        포지션 크기 계산
        
        Parameters:
        -----------
        available_cash : float
            사용 가능 현금
        num_stocks : int
            매수할 종목 수
        stock_price : float
            주가
        
        Returns:
        --------
        int
            매수 수량
        """
        if num_stocks == 0:
            return 0
        
        allocation = available_cash / num_stocks
        quantity = int(allocation / stock_price)
        
        logger.debug(f"포지션 계산: 현금={available_cash:,.0f}, "
                    f"종목수={num_stocks}, "
                    f"주가={stock_price:,.0f}, "
                    f"수량={quantity}")
        
        return quantity


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    db = DatabaseManager()
    strategy = ETFStrategy(db)
    
    print("\n=== 테스트: 모멘텀 선발 ===")
    top_stocks = strategy.calculate_momentum_score()
    
    if top_stocks:
        print("\n선발된 종목:")
        for code, score, name in top_stocks:
            print(f"  {name} ({code}): {score:.4f}")
    
    db.disconnect()
