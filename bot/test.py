from config_loader import ConfigLoader

loader = ConfigLoader()
print(f"계좌번호: '{loader.get_account_no()}'")
print(f"비밀번호: '{loader.get_account_password()}'")
print(f"자동선택: {loader.get_auto_select()}")

