"""
kiwoom_api.py
키움증권 OpenAPI+ 래퍼 클래스 (타임아웃 추가)
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QAxContainer import QAxWidget
from PyQt5.QtCore import QEventLoop, QTimer
import logging

logger = logging.getLogger(__name__)


class KiwoomAPI:
    """키움증권 OpenAPI+ 래퍼 클래스"""
    
    def __init__(self, account_password=None):
        """초기화"""
        logger.info("키움 API 초기화 시작...")
        
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        
        self.login_event_loop = None
        self.order_event_loop = None
        
        self.account_number = None
        self.account_list = []
        self.account_password = account_password if account_password else ""
        self.order_result = {}
        self.real_data = {}
        self.deposit = 0
        self.balance = {}
        self.current_price = 0
        
        self._connect_events()
        
        logger.info("키움 API 초기화 완료")
    
    def _connect_events(self):
        """이벤트 핸들러 연결"""
        self.ocx.OnEventConnect.connect(self._on_event_connect)
        self.ocx.OnReceiveTrData.connect(self._on_receive_tr_data)
        self.ocx.OnReceiveChejanData.connect(self._on_receive_chejan_data)
        self.ocx.OnReceiveRealData.connect(self._on_receive_real_data)
        self.ocx.OnReceiveMsg.connect(self._on_receive_msg)
    
    def login(self):
        """로그인"""
        logger.info("로그인 시작...")
        
        self.login_event_loop = QEventLoop()
        self.ocx.dynamicCall("CommConnect()")
        
        # 타임아웃 30초
        QTimer.singleShot(30000, self.login_event_loop.quit)
        self.login_event_loop.exec_()
        
        if self.get_connect_state() == 1:
            logger.info("로그인 성공")
            self._get_account_info()
            return True
        else:
            logger.error("로그인 실패")
            return False
    
    def _on_event_connect(self, err_code):
        """로그인 이벤트 핸들러"""
        if err_code == 0:
            logger.info("로그인 이벤트: 성공")
        else:
            logger.error(f"로그인 이벤트: 실패 (코드={err_code})")
        
        if self.login_event_loop:
            self.login_event_loop.exit()
    
    def get_connect_state(self):
        """연결 상태 확인"""
        return self.ocx.dynamicCall("GetConnectState()")
    
    def _get_account_info(self):
        """계좌 정보 조회"""
        account_list = self.ocx.dynamicCall("GetLoginInfo(QString)", "ACCNO")
        self.account_list = account_list.split(';')[:-1]
        
        if self.account_list:
            self.account_number = self.account_list[0]
            logger.info(f"계좌번호: {self.account_number}")
        else:
            logger.error("계좌 정보 없음")
    
    def get_deposit(self):
        """예수금 조회"""
        logger.info("예수금 조회 요청...")
        
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "계좌번호", self.account_number)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "비밀번호", self.account_password)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "비밀번호입력매체구분", "01")  # 00 → 01
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "조회구분", "2")
        
        ret = self.ocx.dynamicCall("CommRqData(QString, QString, int, QString)", 
                                "예수금상세현황요청", "opw00001", 0, "0101")
        
        if ret != 0:
            logger.error(f"예수금 조회 요청 실패: {ret}")
            return 0
        
        logger.info("예수금 조회 요청 전송 완료, 응답 대기 중...")
        
        self.order_event_loop = QEventLoop()
        QTimer.singleShot(10000, self.order_event_loop.quit)
        self.order_event_loop.exec_()
        
        logger.info(f"예수금 조회 완료: {self.deposit:,}원")
        
        return self.deposit

    def get_balance(self):
        """보유 종목 조회"""
        logger.info("잔고 조회 요청...")
        
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "계좌번호", self.account_number)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "비밀번호", self.account_password)
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "비밀번호입력매체구분", "01")  # 00 → 01
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "조회구분", "2")
        
        ret = self.ocx.dynamicCall("CommRqData(QString, QString, int, QString)", 
                                "계좌평가잔고내역요청", "opw00018", 0, "0102")
        
        if ret != 0:
            logger.error(f"잔고 조회 요청 실패: {ret}")
            return {}
        
        logger.info("잔고 조회 요청 전송 완료, 응답 대기 중...")
        
        self.order_event_loop = QEventLoop()
        QTimer.singleShot(10000, self.order_event_loop.quit)
        self.order_event_loop.exec_()
        
        logger.info(f"잔고 조회 완료: {len(self.balance)}개 종목")
        
        return self.balance

    
    def send_order(self, order_type, code, quantity, price=0):
        """주문 전송"""
        hoga_type = "03" if price == 0 else "00"
        screen_no = "0101"
        order_name = "매수" if order_type == "1" else "매도"
        
        logger.info(f"{order_name} 주문: {code}, 수량={quantity}, 가격={price}")
        
        result = self.ocx.dynamicCall(
            "SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
            [order_name, screen_no, self.account_number, int(order_type), 
             code, quantity, price, hoga_type, ""]
        )
        
        if result == 0:
            logger.info("주문 전송 성공")
            
            self.order_event_loop = QEventLoop()
            
            # 타임아웃 10초
            QTimer.singleShot(10000, self.order_event_loop.quit)
            
            self.order_event_loop.exec_()
            
            return self.order_result
        else:
            logger.error(f"주문 전송 실패: 코드={result}")
            return {'success': False, 'code': result}
    
    def buy(self, code, quantity, price=0):
        """매수 주문"""
        return self.send_order("1", code, quantity, price)
    
    def sell(self, code, quantity, price=0):
        """매도 주문"""
        return self.send_order("2", code, quantity, price)
    
    def get_current_price(self, code):
        """현재가 조회"""
        self.ocx.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
        
        ret = self.ocx.dynamicCall("CommRqData(QString, QString, int, QString)", 
                                   "주식기본정보요청", "opt10001", 0, "0103")
        
        if ret != 0:
            logger.error(f"현재가 조회 실패: {ret}")
            return 0
        
        self.order_event_loop = QEventLoop()
        
        # 타임아웃 10초
        QTimer.singleShot(10000, self.order_event_loop.quit)
        
        self.order_event_loop.exec_()
        
        return self.current_price
    
    def _on_receive_tr_data(self, screen_no, rqname, trcode, record_name, 
                           prev_next, data_len, err_code, msg, splm_msg):
        """TR 데이터 수신 이벤트"""
        logger.info(f"TR 데이터 수신: {rqname}")
        
        if rqname == "예수금상세현황요청":
            self._process_deposit_data()
        elif rqname == "계좌평가잔고내역요청":
            self._process_balance_data()
        elif rqname == "주식기본정보요청":
            self._process_current_price_data()
        
        if self.order_event_loop and self.order_event_loop.isRunning():
            self.order_event_loop.exit()
    
    def _process_deposit_data(self):
        """예수금 데이터 처리"""
        try:
            deposit = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                          "opw00001", "예수금상세현황요청", 0, "예수금")
            self.deposit = int(deposit.strip())
            logger.info(f"예수금 파싱 완료: {self.deposit:,}원")
        except Exception as e:
            logger.error(f"예수금 데이터 처리 오류: {e}")
            self.deposit = 0
    
    def _process_balance_data(self):
        """잔고 데이터 처리"""
        try:
            self.balance = {}
            
            count = self.ocx.dynamicCall("GetRepeatCnt(QString, QString)",
                                         "opw00018", "계좌평가잔고내역요청")
            
            logger.info(f"보유 종목 수: {count}개")
            
            for i in range(count):
                code = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                           "opw00018", "계좌평가잔고내역요청", i, "종목번호").strip()
                
                name = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                           "opw00018", "계좌평가잔고내역요청", i, "종목명").strip()
                
                quantity = int(self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                                   "opw00018", "계좌평가잔고내역요청", i, "보유수량").strip())
                
                price = abs(int(self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                                     "opw00018", "계좌평가잔고내역요청", i, "현재가").strip()))
                
                buy_price = abs(int(self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                                         "opw00018", "계좌평가잔고내역요청", i, "매입가").strip()))
                
                self.balance[code] = {
                    'name': name,
                    'quantity': quantity,
                    'price': price,
                    'buy_price': buy_price
                }
                
                logger.info(f"  - {name} ({code}): {quantity}주")
            
        except Exception as e:
            logger.error(f"잔고 데이터 처리 오류: {e}")
            self.balance = {}
    
    def _process_current_price_data(self):
        """현재가 데이터 처리"""
        try:
            price = self.ocx.dynamicCall("GetCommData(QString, QString, int, QString)",
                                        "opt10001", "주식기본정보요청", 0, "현재가")
            self.current_price = abs(int(price.strip()))
        except Exception as e:
            logger.error(f"현재가 데이터 처리 오류: {e}")
            self.current_price = 0
    
    def _on_receive_chejan_data(self, gubun, item_cnt, fid_list):
        """체결 데이터 수신 이벤트"""
        if gubun == "0":
            order_no = self.ocx.dynamicCall("GetChejanData(int)", 9203)
            code = self.ocx.dynamicCall("GetChejanData(int)", 9001)
            order_status = self.ocx.dynamicCall("GetChejanData(int)", 913)
            
            logger.info(f"체결 통보: 주문번호={order_no}, 종목={code}, 상태={order_status}")
            
            self.order_result = {
                'success': True,
                'order_no': order_no,
                'code': code,
                'status': order_status
            }
            
            if self.order_event_loop and self.order_event_loop.isRunning():
                self.order_event_loop.exit()
    
    def _on_receive_real_data(self, code, real_type, real_data):
        """실시간 데이터 수신 이벤트"""
        pass
    
    def _on_receive_msg(self, screen_no, rqname, trcode, msg):
        """메시지 수신 이벤트"""
        logger.info(f"메시지: {msg}")


if __name__ == "__main__":
    import getpass
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    password = getpass.getpass("계좌 비밀번호 (4자리, 엔터=빈값): ")
    if not password:
        password = ""
    
    api = KiwoomAPI(account_password=password)
    
    if api.login():
        print("\n=== 로그인 성공 ===")
        print(f"계좌번호: {api.account_number}")
        
        print("\n=== 예수금 조회 ===")
        deposit = api.get_deposit()
        print(f"예수금: {deposit:,}원")
        
        print("\n=== 잔고 조회 ===")
        balance = api.get_balance()
        if balance:
            for code, info in balance.items():
                print(f"{info['name']} ({code}): {info['quantity']}주, {info['price']:,}원")
        else:
            print("보유 종목 없음")
    
    sys.exit(api.app.exec_())

