# -*- coding: utf-8 -*-
"""
[전체 전략 그리드 백테스트] + [슬리피지 0.05/0.10/0.20% 3개] + [연도별 수익률/연도별 MDD]
(2016~2025 와이드 테이블 저장 버전)
===============================================================================================

이 스크립트의 목적
------------------
1) 전략 그리드(수천~수만 조합)를 모두 백테스트한다.
2) 각 전략에 대해 슬리피지 3개(0.05%, 0.10%, 0.20%)를 각각 적용한 성과를 계산한다.
3) 각 슬리피지별로:
   - 전체 기간 요약 지표 (CAGR, MDD, Total Return, trading_days, entry/exit count)
   - 연도별 수익률, 연도별 MDD
   를 “와이드(wide)” 형태(전략 1개 = DB 1행)로 저장한다.

사용자가 확정한 핵심 룰(중요)
----------------------------
A) 평가/선발/필터(S/A/F)는 전날(t-1) 데이터만 사용 (룩어헤드 방지)
B) 청산 신호는 t일 장 마감 후 확정:
   - SMA(20), ATR(14)는 “당일 종가 기준으로 계산된 DB 컬럼값” 사용 OK
   - 단, 실제 매도 체결은 규칙에 따라 '다음날 시가' 또는 '당일 종가'
C) 진입 당일은 청산 평가 제외(최소 1일 보유)
D) E0는 2가지 체결 방식
   - XITE0CLOSE: H 도달 시 '당일 종가' 청산
   - XITE0NEXT_OPEN: H 도달 시 '다음날 시가' 청산
E) E1~E4는 (H 도달 OR 조건 충족) 시 무조건 다음날 시가 청산
   - E1: Close < SMA20
   - E2: Close < Entry - x*ATR
   - E3: Close < High - m*ATR   (High=당일 고가, 확정)
   - E4: (E2 OR E3)
F) 슬리피지 적용:
   - 체결가에만 적용(신호 판단에는 미적용)
   - 롱 기준 불리하게:
       매수 fill = raw * (1 + slip)
       매도 fill = raw * (1 - slip)
   - slip 입력은 % 단위(0.10)지만 내부 계산은 비율(0.001)

성능/운영
---------
- multiprocessing.Pool 병렬 처리
- 워커는 시작 시 kr_etf 데이터를 1회 로딩해서 캐시
- 메인은 chunk_commit 단위로 UPSERT(체크포인트 저장)

주의
----
- 와이드 테이블은 컬럼 수가 많아 1행 INSERT가 무거울 수 있습니다.
  chunk_commit은 100~300 정도를 권장합니다(너무 크게 잡지 마세요).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
import itertools
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import multiprocessing as mp
from tqdm import tqdm


# =====================================================
# 1) DB 설정
# =====================================================
db_user = 'root'
db_pass = '1234'
db_host = 'localhost'
db_port = '3306'
db_schema = 'stock'

source_table = 'kr_etf'

# (중요) 위에서 만든 2016~2025 와이드 테이블명
result_table = 'etf_backtest_wide_slip_2016_2025'


# =====================================================
# 2) 유니버스(12개) - 문자열 유지(앞자리 0 보존)
# =====================================================
UNIVERSE_CODES = [
    '091160', '395160', '305720', '449450', '329200', '494670',
    '445290', '139260', '091180', '091170', '102970', '244580'
]


# =====================================================
# 3) 연도 컬럼 범위 (와이드 테이블 컬럼과 반드시 일치해야 함)
# =====================================================
YEAR_START = 2016
YEAR_END = 2025
YEARS = list(range(YEAR_START, YEAR_END + 1))


# =====================================================
# 4) 슬리피지 시나리오 (% 단위) + 컬럼 접미사(suffix)
# =====================================================
SLIPPAGE_PCTS = [0.05, 0.10, 0.20]

# suffix 규칙:
# - 0.05% -> s005
# - 0.10% -> s010
# - 0.20% -> s020
SLIP_SUFFIX = {
    0.05: "s005",
    0.10: "s010",
    0.20: "s020",
}


# =====================================================
# 5) 전략 그리드(기존과 동일)
# =====================================================
S_CHOICES = ['S0', 'S1', 'S2', 'S3']
N_CHOICES = [1, 2, 3, 4]
A_CHOICES = ['A0', 'A1']
F_CHOICES = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5']

T_CHOICES = ['T1', 'T2', 'T3P', 'T4', 'T5']
K_GRID = [0.4, 0.6, 0.8]  # T1/T2만

H_GRID = [2, 3, 5, 7, 10]
X_GRID = [1.2, 1.5, 1.8]
M_GRID = [2.2, 2.5, 3.0]
E_TYPES = ['E0', 'E1', 'E2', 'E3', 'E4']

E0_EXIT_MODES = ['CLOSE', 'NEXT_OPEN']  # E0만 2개


@dataclass(frozen=True)
class StrategySpec:
    """
    전략 1개를 완전히 정의.
    e0_exit_mode는 E0에만 의미가 있고, E1~E4는 NEXT_OPEN 고정으로 생성(중복 제거).
    """
    s: str
    n: int
    a: str
    f: str
    t: str
    k: Optional[float]
    e: str
    h: int
    x: Optional[float]
    m: Optional[float]
    e0_exit_mode: str

    def strategy_id(self) -> str:
        """
        exit_mode 컬럼이 없으므로 strategy_id에 포함하여 PK를 유일하게 구성.
        예: ...-E0-H2-XITE0CLOSE
            ...-E0-H2-XITE0NEXT_OPEN
        """
        base = f"{self.s}-N{self.n}-{self.a}-{self.f}-{self.t}"
        if self.k is not None:
            base += f"-k{self.k}"
        base += f"-{self.e}-H{self.h}"
        if self.x is not None:
            base += f"-x{self.x}"
        if self.m is not None:
            base += f"-m{self.m}"
        base += f"-XITE0{self.e0_exit_mode}"
        return base


def generate_strategies() -> Iterable[StrategySpec]:
    """
    전략 조합 생성기.

    중복 제거:
    - S0는 랭킹을 쓰지 않으므로 N이 의미 없음 -> n=0 하나만 생성
    - E0만 exit_mode 2개, E1~E4는 의미없으므로 NEXT_OPEN만 생성
    """
    for s in S_CHOICES:
        n_list = [0] if s == 'S0' else N_CHOICES

        for n, a, f, t, e, h in itertools.product(
            n_list, A_CHOICES, F_CHOICES, T_CHOICES, E_TYPES, H_GRID
        ):
            k_list = K_GRID if t in ('T1', 'T2') else [None]
            e0_modes = E0_EXIT_MODES if e == 'E0' else ['NEXT_OPEN']

            for k in k_list:
                for e0_mode in e0_modes:
                    if e == 'E2':
                        for x in X_GRID:
                            yield StrategySpec(s,n,a,f,t,k,e,h,x,None,e0_mode)
                    elif e == 'E3':
                        for m in M_GRID:
                            yield StrategySpec(s,n,a,f,t,k,e,h,None,m,e0_mode)
                    elif e == 'E4':
                        for x in X_GRID:
                            for m in M_GRID:
                                yield StrategySpec(s,n,a,f,t,k,e,h,x,m,e0_mode)
                    else:
                        yield StrategySpec(s,n,a,f,t,k,e,h,None,None,e0_mode)


# =====================================================
# 6) DB 유틸
# =====================================================
def make_engine():
    """SQLAlchemy engine 생성."""
    url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_schema}"
    return create_engine(url, pool_pre_ping=True)


def load_source_data(engine) -> pd.DataFrame:
    """
    원천 테이블에서 유니버스 12개만 로딩.
    - Date datetime
    - Code str
    - 수치 컬럼 numeric
    """
    sql = text(f"""
        SELECT
            Date, Code, Name,
            Open, High, Low, Close, Volume,
            r_12w, r_26w,
            sma_20, sma_60, vma_20, vma_60,
            atr_14
        FROM {source_table}
        WHERE Code IN :codes
        ORDER BY Date ASC
    """)
    df = pd.read_sql(sql, engine, params={"codes": tuple(UNIVERSE_CODES)})

    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)

    num_cols = [
        "Open","High","Low","Close","Volume",
        "r_12w","r_26w",
        "sma_20","sma_60","vma_20","vma_60",
        "atr_14"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =====================================================
# 7) 선발/필터 로직 (평가일: t-1)
# =====================================================
def compute_score(s_choice: str, row_prev: pd.Series) -> float:
    """상대모멘텀 점수."""
    if s_choice == "S1":
        return row_prev["r_12w"]
    if s_choice == "S2":
        return 0.7 * row_prev["r_12w"] + 0.3 * row_prev["r_26w"]
    if s_choice == "S3":
        r12 = row_prev["r_12w"]
        r26 = row_prev["r_26w"]
        if pd.isna(r12) or pd.isna(r26):
            return np.nan
        return 0.7 * np.log1p(r12) + 0.3 * np.log1p(r26)
    return 0.0  # S0


def pass_gate_A(a_choice: str, row_prev: pd.Series) -> bool:
    """절대모멘텀 게이트."""
    if a_choice == "A0":
        return True
    if a_choice == "A1":
        return (not pd.isna(row_prev["r_12w"])) and (row_prev["r_12w"] > 0)
    raise ValueError(a_choice)


def pass_filter_F(f_choice: str, row_prev: pd.Series) -> bool:
    """진입 필터."""
    c = row_prev["Close"]
    sma20, sma60 = row_prev["sma_20"], row_prev["sma_60"]
    vma20, vma60 = row_prev["vma_20"], row_prev["vma_60"]

    if f_choice == "F0":
        return True
    if f_choice == "F1":
        return c > sma60
    if f_choice == "F2":
        return sma20 > sma60
    if f_choice == "F3":
        return vma20 > vma60
    if f_choice == "F4":
        return (c > sma60) and (vma20 > vma60)
    if f_choice == "F5":
        return (sma20 > sma60) and (vma20 > vma60)
    raise ValueError(f_choice)


# =====================================================
# 8) 포지션/진입/청산 (슬리피지 포함)
# =====================================================
@dataclass
class Position:
    """
    1개 종목 포지션.
    entry_price는 '슬리피지 적용된 실제 매수 체결가'를 기록.
    """
    code: str
    entry_idx: int
    entry_price: float


def hold_days(pos: Position, t_idx: int) -> int:
    """진입 다음날=1일. hold_days = t_idx - entry_idx"""
    return t_idx - pos.entry_idx


def try_entry(spec: StrategySpec,
              df_by_code: Dict[str, pd.DataFrame],
              dates: pd.DatetimeIndex,
              t_idx: int,
              selected_codes: List[str],
              slip: float) -> List[Position]:
    """
    진입 체결(일봉 단순화).
    - raw_entry를 계산한 뒤 매수 슬리피지 적용: fill = raw_entry*(1+slip)
    """
    t_date = dates[t_idx]
    out: List[Position] = []

    for code in selected_codes:
        d = df_by_code[code]
        if t_date not in d.index:
            continue

        row_t = d.loc[t_date]
        if pd.isna(row_t["Open"]) or pd.isna(row_t["High"]) or pd.isna(row_t["Close"]):
            continue

        if spec.t in ("T1", "T2", "T3P") and t_idx == 0:
            continue

        raw_entry = None

        # T1: Open_t + k*ATR_{t-1} 돌파
        if spec.t == "T1":
            prev_date = dates[t_idx - 1]
            if prev_date not in d.index:
                continue
            atr_prev = d.loc[prev_date, "atr_14"]
            if pd.isna(atr_prev) or spec.k is None:
                continue
            trigger = float(row_t["Open"]) + spec.k * float(atr_prev)
            if float(row_t["High"]) >= trigger:
                raw_entry = trigger

        # T2: Open_t + k*(High_{t-1}-Low_{t-1}) 돌파
        elif spec.t == "T2":
            prev_date = dates[t_idx - 1]
            if prev_date not in d.index:
                continue
            row_prev = d.loc[prev_date]
            if pd.isna(row_prev["High"]) or pd.isna(row_prev["Low"]) or spec.k is None:
                continue
            trigger = float(row_t["Open"]) + spec.k * (float(row_prev["High"]) - float(row_prev["Low"]))
            if float(row_t["High"]) >= trigger:
                raw_entry = trigger

        # T3P: High_{t-1} 돌파, 체결=max(trigger, Open_t)
        elif spec.t == "T3P":
            prev_date = dates[t_idx - 1]
            if prev_date not in d.index:
                continue
            prev_high = d.loc[prev_date, "High"]
            if pd.isna(prev_high):
                continue
            trigger = float(prev_high)
            if float(row_t["High"]) >= trigger:
                raw_entry = float(max(trigger, float(row_t["Open"])))

        # T4: 시가 매수
        elif spec.t == "T4":
            raw_entry = float(row_t["Open"])

        # T5: 종가 매수
        elif spec.t == "T5":
            raw_entry = float(row_t["Close"])

        else:
            raise ValueError(spec.t)

        if raw_entry is None:
            continue

        fill_buy = float(raw_entry) * (1.0 + slip)
        out.append(Position(code=code, entry_idx=t_idx, entry_price=fill_buy))

    return out


def exit_signal(spec: StrategySpec,
                pos: Position,
                d: pd.DataFrame,
                t_date: pd.Timestamp,
                t_idx: int) -> Tuple[bool, str]:
    """
    t일 종가 기준으로 청산 신호 평가.

    action:
      - "CLOSE_TODAY"   : 당일 종가 청산(E0 + CLOSE)
      - "PEND_NEXTOPEN" : 다음날 시가 청산(E0+NEXT_OPEN, E1~E4)

    중요:
    - 진입 당일은 청산 평가 제외
    - SMA/ATR은 t일 값 사용(당일 종가 기준 지표)
    """
    if t_idx == pos.entry_idx:
        return False, ""

    if t_date not in d.index:
        return False, ""
    row = d.loc[t_date]

    needed = ["Close", "High", "sma_20", "atr_14"]
    if any(pd.isna(row.get(c, np.nan)) for c in needed):
        return False, ""

    close_t = float(row["Close"])
    high_t = float(row["High"])
    sma20_t = float(row["sma_20"])
    atr_t = float(row["atr_14"])

    h_hit = (hold_days(pos, t_idx) >= spec.h)

    # E0: H 도달만 체크
    if spec.e == "E0":
        if not h_hit:
            return False, ""
        if spec.e0_exit_mode == "CLOSE":
            return True, "CLOSE_TODAY"
        elif spec.e0_exit_mode == "NEXT_OPEN":
            return True, "PEND_NEXTOPEN"
        else:
            raise ValueError(spec.e0_exit_mode)

    # E1~E4: (H or 조건) => NEXT_OPEN
    cond = False

    if spec.e == "E1":
        cond = (close_t < sma20_t)

    elif spec.e == "E2":
        if spec.x is not None:
            # Entry는 "실제 매수 체결가(슬리피지 포함)"를 쓰는 것이 일관됨
            cond = (close_t < (pos.entry_price - spec.x * atr_t))

    elif spec.e == "E3":
        if spec.m is not None:
            # 확정: 당일 고가 기반
            cond = (close_t < (high_t - spec.m * atr_t))

    elif spec.e == "E4":
        c2 = False
        c3 = False
        if spec.x is not None:
            c2 = (close_t < (pos.entry_price - spec.x * atr_t))
        if spec.m is not None:
            c3 = (close_t < (high_t - spec.m * atr_t))
        cond = (c2 or c3)

    else:
        raise ValueError(spec.e)

    if h_hit or cond:
        return True, "PEND_NEXTOPEN"
    return False, ""


# =====================================================
# 9) 단일 전략 + 단일 슬리피지 백테스트
# =====================================================
def backtest_one_slip(spec: StrategySpec,
                      df_all: pd.DataFrame,
                      start: str,
                      end: str,
                      slip_pct: float,
                      force_liquidate_end: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    slip_pct는 % 단위(0.10)로 받고, 내부에서는 비율(0.001)로 변환한다.
    반환:
      - eq: Date index, Equity (종가 기준)
      - summary: 전체 기간 요약지표(소수)
    """
    slip = float(slip_pct) / 100.0

    df_by_code = {code: g.sort_values("Date").set_index("Date") for code, g in df_all.groupby("Code")}

    dates = pd.DatetimeIndex(sorted(df_all["Date"].unique()))
    dates = dates[(dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))]
    if len(dates) < 2:
        raise ValueError("Not enough dates in range")

    cash = 1.0
    positions: List[Position] = []
    pos_principal: Dict[str, float] = {}  # code -> 원금
    pending_exit_codes: set = set()

    entry_count = 0
    exit_count = 0
    equity_curve = []

    for t_idx, t_date in enumerate(dates):
        # (1) 장 시작: pending 포지션을 오늘 시가로 매도 체결(매도 슬리피지 적용)
        if pending_exit_codes and positions:
            to_close = [p for p in positions if p.code in pending_exit_codes]
            still_open = [p for p in positions if p.code not in pending_exit_codes]

            for pos in to_close:
                d = df_by_code[pos.code]
                if t_date not in d.index or pd.isna(d.loc[t_date, "Open"]):
                    still_open.append(pos)
                    continue

                raw_open = float(d.loc[t_date, "Open"])
                fill_sell = raw_open * (1.0 - slip)

                principal = pos_principal.get(pos.code, 0.0)
                cash += principal * (fill_sell / pos.entry_price)

                pos_principal.pop(pos.code, None)
                pending_exit_codes.discard(pos.code)
                exit_count += 1

            positions = still_open

        # (2) 진입: 포지션이 없을 때만. 전날(t-1) 평가로 후보 선정 후 오늘 진입 시도
        if (not positions) and (t_idx > 0):
            eval_date = dates[t_idx - 1]

            candidates = []
            needed_prev = ["Close","r_12w","r_26w","sma_20","sma_60","vma_20","vma_60","atr_14"]

            for code, d in df_by_code.items():
                if eval_date not in d.index:
                    continue
                row_prev = d.loc[eval_date]

                # 전일 지표 결측이면 후보 제외(동적 유니버스)
                if any(pd.isna(row_prev.get(col, np.nan)) for col in needed_prev):
                    continue

                if not pass_gate_A(spec.a, row_prev):
                    continue
                if not pass_filter_F(spec.f, row_prev):
                    continue

                candidates.append((code, row_prev))

            if candidates:
                if spec.s == "S0":
                    selected = [c for c, _ in candidates]
                else:
                    scored = []
                    for code, row_prev in candidates:
                        sc = compute_score(spec.s, row_prev)
                        if not pd.isna(sc):
                            scored.append((code, sc))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    selected = [c for c, _ in scored[:spec.n]]

                new_positions = try_entry(spec, df_by_code, dates, t_idx, selected, slip=slip)
                if new_positions:
                    n_actual = len(new_positions)
                    alloc = cash / n_actual
                    cash = 0.0
                    positions = new_positions
                    pos_principal = {p.code: alloc for p in positions}
                    entry_count += n_actual

        # (3) 장 마감: 청산 신호 생성
        if positions:
            closed_now = []

            for pos in positions:
                d = df_by_code[pos.code]
                hit, action = exit_signal(spec, pos, d, t_date, t_idx)
                if not hit:
                    continue

                if action == "CLOSE_TODAY":
                    # 당일 종가 매도 체결 + 매도 슬리피지 적용
                    if t_date not in d.index or pd.isna(d.loc[t_date, "Close"]):
                        continue
                    raw_close = float(d.loc[t_date, "Close"])
                    fill_sell = raw_close * (1.0 - slip)

                    principal = pos_principal.get(pos.code, 0.0)
                    cash += principal * (fill_sell / pos.entry_price)

                    pos_principal.pop(pos.code, None)
                    closed_now.append(pos.code)
                    exit_count += 1

                elif action == "PEND_NEXTOPEN":
                    pending_exit_codes.add(pos.code)

                else:
                    raise ValueError(action)

            if closed_now:
                positions = [p for p in positions if p.code not in closed_now]

        # (4) 종가 기준 Equity 평가(표시용) - 체결이 아니므로 슬리피지 없음
        pos_value = 0.0
        for pos in positions:
            d = df_by_code[pos.code]
            if t_date not in d.index or pd.isna(d.loc[t_date, "Close"]):
                continue
            close_t = float(d.loc[t_date, "Close"])
            principal = pos_principal.get(pos.code, 0.0)
            pos_value += principal * (close_t / pos.entry_price)

        equity_curve.append((t_date, cash + pos_value))

    # (5) 종료 강제 청산(리포팅 목적): 마지막 종가로 매도(+슬리피지)
    if force_liquidate_end and positions:
        last_date = dates[-1]
        for pos in positions:
            d = df_by_code[pos.code]
            if last_date in d.index and not pd.isna(d.loc[last_date, "Close"]):
                raw_close = float(d.loc[last_date, "Close"])
                fill_sell = raw_close * (1.0 - slip)
                principal = pos_principal.get(pos.code, 0.0)
                cash += principal * (fill_sell / pos.entry_price)
                exit_count += 1
        equity_curve[-1] = (last_date, cash)

    eq = pd.DataFrame(equity_curve, columns=["Date","Equity"]).set_index("Date")

    total_return = float(eq["Equity"].iloc[-1] / eq["Equity"].iloc[0] - 1.0)
    peak = eq["Equity"].cummax()
    dd = eq["Equity"] / peak - 1.0
    mdd = float(dd.min())

    trading_days = int(len(eq))
    years = trading_days / 252.0
    cagr = float((eq["Equity"].iloc[-1] / eq["Equity"].iloc[0]) ** (1/years) - 1.0) if years > 0 else np.nan

    summary = {
        "cagr": cagr,
        "mdd": mdd,
        "total_return": total_return,
        "trading_days": trading_days,
        "entry_count": int(entry_count),
        "exit_count": int(exit_count),
        "round_trips": int(exit_count),
    }
    return eq, summary


def yearly_from_equity(eq: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """
    전체 기간 Equity로부터 연도별 수익률/연도별 MDD 계산.
    반환: {year: {"yearly_return": 소수, "yearly_mdd": 소수}}
    """
    tmp = eq.copy()
    tmp["Year"] = tmp.index.year

    out: Dict[int, Dict[str, float]] = {}
    for y, g in tmp.groupby("Year"):
        g = g.sort_index()
        y_ret = float(g["Equity"].iloc[-1] / g["Equity"].iloc[0] - 1.0)
        peak = g["Equity"].cummax()
        dd = g["Equity"] / peak - 1.0
        y_mdd = float(dd.min())
        out[int(y)] = {"yearly_return": y_ret, "yearly_mdd": y_mdd}

    return out


# =====================================================
# 10) 전략 1개 -> 와이드 1행 dict 생성
# =====================================================
def run_one_spec_wide(spec: StrategySpec, df_all: pd.DataFrame, start: str, end: str) -> Dict:
    """
    전략 1개에 대해 3개 슬리피지를 모두 계산해 “한 행”으로 합친 dict를 만든다.
    dict의 키 = DB 컬럼명과 일치해야 UPSERT 가능.
    """
    sid = spec.strategy_id()

    row = {
        "strategy_id": sid,
        "s": spec.s, "n": int(spec.n), "a": spec.a, "f": spec.f,
        "t": spec.t, "k": None if spec.k is None else float(spec.k),
        "e": spec.e, "h": int(spec.h),
        "x": None if spec.x is None else float(spec.x),
        "m": None if spec.m is None else float(spec.m),
        "start_date": start,
        "end_date": end,
    }

    for slip_pct in SLIPPAGE_PCTS:
        suffix = SLIP_SUFFIX[slip_pct]

        eq, summ = backtest_one_slip(spec, df_all, start, end, slip_pct, force_liquidate_end=True)
        yr = yearly_from_equity(eq)

        # 요약(소수 -> % 저장)
        row[f"cagr_{suffix}"] = None if pd.isna(summ["cagr"]) else round(float(summ["cagr"]) * 100.0, 2)
        row[f"mdd_{suffix}"] = round(float(summ["mdd"]) * 100.0, 2)
        row[f"total_return_{suffix}"] = round(float(summ["total_return"]) * 100.0, 2)
        row[f"trading_days_{suffix}"] = int(summ["trading_days"])
        row[f"entry_count_{suffix}"] = int(summ["entry_count"])
        row[f"exit_count_{suffix}"] = int(summ["exit_count"])
        row[f"round_trips_{suffix}"] = int(summ["round_trips"])

        # 연도별(없는 연도는 NULL)
        for y in YEARS:
            if y in yr:
                row[f"yearly_return_{y}_{suffix}"] = round(float(yr[y]["yearly_return"]) * 100.0, 2)
                row[f"yearly_mdd_{y}_{suffix}"] = round(float(yr[y]["yearly_mdd"]) * 100.0, 2)
            else:
                row[f"yearly_return_{y}_{suffix}"] = None
                row[f"yearly_mdd_{y}_{suffix}"] = None

    return row


# =====================================================
# 11) 병렬 워커 캐시
# =====================================================
_G_DF_ALL = None
_G_START = None
_G_END = None

def _worker_init(start: str, end: str):
    """워커 시작 시 1회: DB에서 원천 데이터 로딩 -> 전역 캐시"""
    global _G_DF_ALL, _G_START, _G_END
    _G_START = start
    _G_END = end
    engine = make_engine()
    _G_DF_ALL = load_source_data(engine)

def _worker_run(spec: StrategySpec) -> Dict:
    """워커: 전략 1개 실행 -> 와이드 1행 dict 반환"""
    global _G_DF_ALL, _G_START, _G_END
    try:
        return run_one_spec_wide(spec, _G_DF_ALL, _G_START, _G_END)
    except Exception as e:
        return {"strategy_id": spec.strategy_id(), "error": str(e)}


# =====================================================
# 12) DB UPSERT 유틸
# =====================================================
def build_column_list() -> List[str]:
    """
    INSERT/UPSERT에 사용할 컬럼 리스트를 “테이블 정의와 동일하게” 구성.
    - 컬럼명이 조금이라도 다르면 INSERT에서 오류가 납니다.
    """
    cols = [
        "strategy_id",
        "s","n","a","f","t","k","e","h","x","m",
        "start_date","end_date",
    ]

    for slip_pct in SLIPPAGE_PCTS:
        suffix = SLIP_SUFFIX[slip_pct]
        cols += [
            f"cagr_{suffix}",
            f"mdd_{suffix}",
            f"total_return_{suffix}",
            f"trading_days_{suffix}",
            f"entry_count_{suffix}",
            f"exit_count_{suffix}",
            f"round_trips_{suffix}",
        ]
        for y in YEARS:
            cols += [
                f"yearly_return_{y}_{suffix}",
                f"yearly_mdd_{y}_{suffix}",
            ]
    return cols


def upsert_wide_rows(engine, rows: List[Dict], cols: List[str]):
    """
    rows(list of dict)를 와이드 테이블에 벌크 UPSERT.
    - error row는 제외
    - PK(strategy_id) 기준 갱신
    """
    clean = [r for r in rows if "error" not in r]
    if not clean:
        return

    # UPDATE 구문에서 PK는 제외
    update_clause = ",".join([f"{c}=VALUES({c})" for c in cols if c != "strategy_id"])

    insert_sql = f"""
    INSERT INTO {result_table} ({",".join(cols)})
    VALUES ({",".join(["%s"] * len(cols))})
    ON DUPLICATE KEY UPDATE {update_clause};
    """

    values = [tuple(r.get(c) for c in cols) for r in clean]

    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.executemany(insert_sql, values)
        conn.commit()
    finally:
        conn.close()


# =====================================================
# 13) 메인 실행(병렬)
# =====================================================
def run_parallel_wide(start: str,
                      end: str,
                      n_workers: Optional[int] = None,
                      chunk_commit: int = 200,
                      limit: Optional[int] = None):
    """
    start/end: 백테스트 기간
    - 와이드 테이블은 YEAR_START~YEAR_END 연도 컬럼이 고정이므로,
      start/end는 최소한 그 범위를 포함하는 기간으로 두는 것이 일반적입니다.
      (여기서는 사용자 요청: 2016~2025)

    chunk_commit:
      - 와이드 1행이 크기 때문에 200 전후 권장
    """
    engine = make_engine()
    cols = build_column_list()

    strategies = list(generate_strategies())
    if limit is not None:
        strategies = strategies[:limit]

    print(f"Total strategies: {len(strategies):,}")
    print(f"Wide years: {YEAR_START}~{YEAR_END} ({len(YEARS)} years)")
    print(f"Slippage(%): {SLIPPAGE_PCTS}")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Workers: {n_workers}")

    buffer: List[Dict] = []

    with mp.Pool(processes=n_workers, initializer=_worker_init, initargs=(start, end)) as pool:
        for res in tqdm(pool.imap_unordered(_worker_run, strategies, chunksize=20),
                        total=len(strategies),
                        desc="Backtesting wide 2016-2025"):

            buffer.append(res)

            if len(buffer) >= chunk_commit:
                upsert_wide_rows(engine, buffer, cols)
                buffer.clear()

        if buffer:
            upsert_wide_rows(engine, buffer, cols)
            buffer.clear()

    print("Done.")


if __name__ == "__main__":
    # 1) 먼저 limit=50~200으로 스키마/UPSERT 정상 동작 확인 권장
    # 2) 문제 없으면 limit=None으로 전체 실행

    # run_parallel_wide(
    #     start="2016-01-01",
    #     end="2025-12-31",
    #     n_workers=8,
    #     chunk_commit=200,
    #     limit=200
    # )

    #전체 실행:
    run_parallel_wide(
        start="2016-01-01",
        end="2025-12-31",
        n_workers=12,
        chunk_commit=200,
        limit=None
    )
