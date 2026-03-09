"""
utils.py
유틸리티 함수 모음
"""

from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_business_days(start_date, end_date):
    """
    두 날짜 사이의 영업일 수 계산 (주말 제외)
    
    Parameters:
    -----------
    start_date : datetime.date or str
        시작일 (YYYY-MM-DD)
    end_date : datetime.date or str
        종료일 (YYYY-MM-DD)
    
    Returns:
    --------
    int
        영업일 수
    """
    # 문자열이면 datetime으로 변환
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # 날짜 순서 확인
    if start_date > end_date:
        return 0
    
    # 영업일 계산
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        # 주말 제외 (0=월요일, 6=일요일)
        if current_date.weekday() < 5:
            business_days += 1
        current_date += timedelta(days=1)
    
    # 시작일 제외 (진입일은 카운트 안 함)
    if business_days > 0:
        business_days -= 1
    
    return business_days


def is_business_day(date):
    """
    영업일 여부 확인
    
    Parameters:
    -----------
    date : datetime.date or str
        확인할 날짜
    
    Returns:
    --------
    bool
        영업일이면 True
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    
    # 주말 체크
    if date.weekday() >= 5:
        return False
    
    # TODO: 공휴일 체크 (필요시 추가)
    # holidays = ['2026-01-01', '2026-03-01', ...]
    # if date.strftime('%Y-%m-%d') in holidays:
    #     return False
    
    return True


if __name__ == "__main__":
    # 테스트
    print("영업일 계산 테스트:")
    
    # 예시 1: 목요일 → 일요일 (3일)
    days = get_business_days('2026-03-06', '2026-03-09')
    print(f"2026-03-06 (목) → 2026-03-09 (일): {days}일")  # 1일 (금요일만)
    
    # 예시 2: 목요일 → 다음주 월요일 (5일)
    days = get_business_days('2026-03-06', '2026-03-10')
    print(f"2026-03-06 (목) → 2026-03-10 (월): {days}일")  # 2일 (금, 월)
    
    # 예시 3: 목요일 → 다음주 화요일 (6일)
    days = get_business_days('2026-03-06', '2026-03-11')
    print(f"2026-03-06 (목) → 2026-03-11 (화): {days}일")  # 3일 (금, 월, 화)
