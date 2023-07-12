from flask import Flask, request, json, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

today = (datetime.now().today()).strftime("%Y-%m-%d")

@app.route("/user-login-info")
def user_login_info():
    uid = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    return "uid123456의 로그인 정보 : 2023/06/22, 2023/06/23, 2023/06/24, 2023/06/25, 2023/06/26"
    
@app.route("/user-buy-item-info")
def user_buy_item_info():
    uid = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return "uid123456의 아이템 구매 정보 : 1. 2023/06/22, 아이템코드 : AA111, 아이템 이름 : HP회복 물약, 가격 : 2000골드 2. 2023/06/23, 아이템코드 : BB222, 아이템 이름 : 롱소드, 가격 : 40000골드"
    
@app.route("/user-cash-charge-info")
def user_cash_charge_info():
    uid = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return "uid123456의 현금 충전 정보 : 1. 2023/06/22, 충전 이름 : A패키지 가격 : 1000원 2. 2023/06/23, 충전 이름 : B패키지, 가격 : 2000원"

    
@app.route("/user-buy-cash-item-info")
def user_buy_cash_item_info():
    uid = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return "uid123456의 현금 아이템 구매 정보 : 1. 2023/06/22, 아이템코드 : CA111, 아이템 이름 : 부활 물약, 가격 : 100캐쉬 2. 2023/06/23, 아이템코드 : CA112, 아이템 이름 : 강화 확률 업 쿠폰, 가격 : 200캐쉬"
    
    
@app.route("/user-refund-cash-item-info")
def user_refund_cash_item_info():
    uid = request.args.get('user_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    return "uid123456의 현금 아이템 환불 정보 : 1. 2023/06/22, 아이템코드 : CA111, 아이템 이름 : 부활 물약, 가격 : 100캐쉬 2. 2023/06/23, 아이템코드 : CA112, 아이템 이름 : 강화 확률 업 쿠폰, 가격 : 200캐쉬"


@app.route("/yesterday-total-login-count")
def yesterday_total_login_count():
    
    yesterday = (datetime.now().today() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    return "어제(" + yesterday + ") " + "접속자 수 : " + "100"

@app.route("/last-month-total-login-count")
def last_month_total_login_count():
    
    data = ""
    for idx in range(1,35):
        data = data + (datetime.now().today() - timedelta(days=idx)).strftime("%Y-%m-%d") + "의 하루 로그인 숫자 = " + str(idx*100) + "\n"
    print(data)

    return "오늘 날짜는 " + today + "이고, " + " 저번달 접속자 수 : " + data

@app.route("/last-year-monthly-login-count")
def last_year_monthly_login_count():
    
    return "No Data"

@app.route("/last-week-daily-sales")
def last_week_daily_sales():
    
    data = ""
    for idx in range(1,10):
        data = data + (datetime.now().today() - timedelta(days=idx)).strftime("%Y-%m-%d") + "의 하루 매출 = " + str(idx*10000) + "\n"
    print(data)

    return "오늘 날짜는 " + today + "이고, " + "최근 1주일 매출 : " + data

@app.route("/last-month-daily-sale")
def last_month_daily_sales():
    
    data = ""
    for idx in range(1,35):
        data = data + (datetime.now().today() - timedelta(days=idx)).strftime("%Y-%m-%d") + "의 하루 매출 = " + str(idx*10000) + "\n"
    print(data)

    return "오늘 날짜는 " + today + "이고, " + "저번달 1달 매출 : " + data
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
