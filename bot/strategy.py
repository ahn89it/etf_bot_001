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
        매수 신호 확인 (래리 윌리엄스 변동성 돌파 전략)
        
        Parameters:
        -----------
        code : str
            종목코드
        current_price : float, optional
            현재가 (None이면 최신 종가 사용)
        
        Returns:
        --------
        dict or None
            매수 신호 정보
        """
        try:
            # 오늘 데이터 조회
            today_str = datetime.now().strftime('%Y-%m-%d')
            df = self.db.get_etf_data(code, date=today_str, days_back=1)
            
            if df.empty:
                logger.warning(f"{code}: 오늘 데이터 없음")
                return None
            
            today = df.iloc[0]
            
            # 필수 데이터 확인
            if pd.isna(today['Open']) or pd.isna(today['Low']):
                logger.warning(f"{code}: 시가/저가 데이터 없음")
                return None
            
            # Trigger 계산
            today_range = today['Open'] - today['Low']
            trigger_price_raw = today['Open'] - (self.k * today_range)
            
            # ★★★ 호가 단위 조정 ★★★
            trigger_price = self.adjust_price_to_tick(trigger_price_raw)
            
            # 현재가 (없으면 최신 종가 사용)
            if current_price is None:
                if pd.isna(today['Close']):
                    logger.warning(f"{code}: 종가 데이터 없음")
                    return None
                current_price = today['Close']
            
            # 매수 조건: 현재가 >= Trigger
            signal = current_price >= trigger_price
            
            result = {
                'signal': signal,
                'trigger_price': trigger_price,
                'current_price': current_price,
                'today_range': today_range,
                'today_open': today['Open'],
                'today_low': today['Low'],
                'today_high': today['High'] if not pd.isna(today['High']) else None,
                'k': self.k
            }
            
            # 로그 출력
            name = ETF_UNIVERSE.get(code, {}).get('name', code)
            logger.info(f"{name} ({code}) 매수 신호 체크:")
            logger.info(f"  시가: {today['Open']:,.0f}원")
            logger.info(f"  저가: {today['Low']:,.0f}원")
            logger.info(f"  Range: {today_range:,.0f}원")
            logger.info(f"  k: {self.k}")
            logger.info(f"  Trigger (원본): {trigger_price_raw:,.2f}원")
            logger.info(f"  Trigger (조정): {trigger_price:,.0f}원")
            logger.info(f"  현재가: {current_price:,.0f}원")
            logger.info(f"  신호: {'✅ 매수' if signal else '❌ 대기'}")
            
            return result
            
        except Exception as e:
            logger.error(f"{code} 매수 신호 확인 오류: {e}")
            return None

    
    def check_sell_signal(self, code, position, hold_days, current_price=None):
        """
        매도 신호 확인
        
        Parameters:
        -----------
        code : str
            종목코드
        position : dict
            포지션 정보
        hold_days : int
            보유일수 (trader.py에서 계산해서 전달)
        current_price : float, optional
            현재가
        
        Returns:
        --------
        dict or None
            매도 신호 정보
        """
        try:
            # 오늘 데이터 조회
            today_str = datetime.now().strftime('%Y-%m-%d')
            df = self.db.get_etf_data(code, date=today_str, days_back=1)
            
            if df.empty:
                logger.warning(f"{code}: 오늘 데이터 없음")
                return None
            
            today = df.iloc[0]
            
            # 매수 당일 체크
            if position['entry_date'] == today_str:
                logger.debug(f"{code}: 매수 당일")
                return {'signal': False, 'reason': '매수 당일'}
            
            # 현재가
            if current_price is None:
                if pd.isna(today['Close']):
                    logger.warning(f"{code}: 종가 데이터 없음")
                    return None
                current_price = today['Close']
            
            # ★★★ 조건 1: 손절 (SMA20 × 0.98 이탈) ★★★
            if not pd.isna(today['sma_20']):
                sma20_threshold = today['sma_20'] * self.sma_buffer
                
                if current_price < sma20_threshold:
                    name = ETF_UNIVERSE.get(code, {}).get('name', code)
                    logger.info(f"{name} ({code}) 손절 신호!")
                    logger.info(f"  현재가: {current_price:,.0f}원")
                    logger.info(f"  SMA(20): {today['sma_20']:,.0f}원")
                    logger.info(f"  손절선: {sma20_threshold:,.0f}원")
                    
                    return {
                        'signal': True,
                        'reason': 'SMA(20) 손절',
                        'sell_price': current_price,
                        'sma20': today['sma_20'],
                        'threshold': sma20_threshold
                    }
            
            # ★★★ 조건 2: 보유기간 도달 ★★★
            if hold_days >= self.hold_days:
                name = ETF_UNIVERSE.get(code, {}).get('name', code)
                logger.info(f"{name} ({code}) 보유기간 도달!")
                logger.info(f"  보유일: {hold_days}일")
                logger.info(f"  최대 보유일: {self.hold_days}일")
                
                return {
                    'signal': True,
                    'reason': f"{self.hold_days}일 도달",
                    'sell_price': current_price,
                    'hold_days': hold_days
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
        if num_stocks == 0 or stock_price == 0:
            return 0
        
        allocation = available_cash / num_stocks
        quantity = int(allocation / stock_price)
        
        logger.debug(f"포지션 계산: 현금={available_cash:,.0f}, "
                    f"종목수={num_stocks}, "
                    f"주가={stock_price:,.0f}, "
                    f"수량={quantity}")
        
        return quantity


    def adjust_price_to_tick(self, price):
        """
        호가 단위에 맞게 가격 조정
        
        Parameters:
        -----------
        price : float
            원본 가격
        
        Returns:
        --------
        int
            호가 단위에 맞춘 가격
        """
        if price < 1000:
            tick = 1
        elif price < 5000:
            tick = 5
        elif price < 10000:
            tick = 10
        elif price < 50000:
            tick = 50
        elif price < 100000:
            tick = 100
        elif price < 500000:
            tick = 500
        else:
            tick = 1000
        
        # 호가 단위로 내림
        adjusted = int(price / tick) * tick
        
        logger.debug(f"가격 조정: {price:,.0f}원 → {adjusted:,.0f}원 (호가단위: {tick}원)")
        
        return adjusted
    

