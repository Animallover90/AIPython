from flask import Flask, request, json, jsonify
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route("/user-login-info", methods=['POST'])
def user_login_info():
    params = request.get_json()
    uid = params.get('uid')
    print(uid)
    
    ## 유저의 uid를 이용해 유저 로그인 정보 반환
    data = [{"login_date" : "2023/06/22"}, {"login_date" : "2023/06/23"}, {"login_date" : "2023/06/24"}, {"login_date" : "2023/06/25"}, {"login_date" : "2023/06/26"}, {"login_date" : "2023/06/27"}]

    response = {
        "result": data
    }

    return jsonify(response)
    
@app.route("/user-buy-item-info", methods=['POST'])
def user_buy_item_info():
    params = request.get_json()
    uid = params.get('uid')
    print(uid)
    
    ## 유저의 uid를 이용해 아이템 구매 정보를 구해서 반환
    data = [{"date" : "2023/06/22", "item_code" : "AA111", "item_name" : "HP회복 물약", "price" : "2000"}, {"date" : "2023/06/23", "item_code" : "BB222", "item_name" : "롱소드", "price" : "40000"}]

    response = {
        "result": data
    }

    return jsonify(response)
    
@app.route("/user-cash-charge-info", methods=['POST'])
def user_cash_charge_info():
    params = request.get_json()
    uid = params.get('uid')
    print(uid)
    
    ## 유저의 uid를 이용해 충전 정보를 구해서 반환
    data = [{"date" : "2023/06/22", "charge_name" : "A패키지", "price" : "1000"}, {"date" : "2023/06/23", "charge_name" : "B패키지", "price" : "2000"}]

    response = {
        "result": data
    }

    return jsonify(response)
    
@app.route("/user-buy-cash-item-info", methods=['POST'])
def user_buy_cash_item_info():
    params = request.get_json()
    uid = params.get('uid')
    print(uid)
    
    ## 유저의 uid를 이용해 캐쉬 아이템 구매 정보를 구해서 반환
    data = [{"date" : "2023/06/22", "cash_code" : "CA111", "cash_name" : "부활 물약", "price" : "100"}, {"date" : "2023/06/23", "cash_code" : "CA112", "cash_name" : "강화 확률 업 쿠폰", "price" : "200"}]

    response = {
        "result": data
    }

    return jsonify(response)
    
    
@app.route("/user-refund-cash-item-info", methods=['POST'])
def user_refund_cash_item_info():
    params = request.get_json()
    uid = params.get('uid')
    print(uid)
    
    ## 유저의 uid를 이용해 캐쉬 아이템 환불 정보를 구해서 반환
    data = [{"date" : "2023/06/22", "cash_code" : "CA111", "cash_name" : "부활 물약", "price" : "100"}, {"date" : "2023/06/23", "cash_code" : "CA112", "cash_name" : "강화 확률 업 쿠폰", "price" : "200"}]

    response = {
        "result": data
    }

    return jsonify(response)

@app.route("/yesterday-total-login-count", methods=['POST'])
def yesterday_total_login_count():
    
    ## 어제 유저의 로그인 숫자
    data = [{"date" : (datetime.now().today() - timedelta(days=1)).strftime("%Y-%m-%d"), "login_count" : "100"}]
    
    response = {
        "result": data
    }

    return jsonify(response)

@app.route("/last-month-total-login-count", methods=['POST'])
def last_month_total_login_count():
    
    ## 최근 한달간의 로그인 정보 반환
    data = list()
    for idx in range(1,35):
        day_data = {"date" : (datetime.now().today() - timedelta(days=idx)).strftime("%Y-%m-%d"), "login_count" : idx*100}
        data.append(day_data)

    response = {
        "result": data
    }

    return jsonify(response)

@app.route("/last-year-monthly-login-count", methods=['POST'])
def last_year_monthly_login_count():
    
    ## 유저의 uid를 이용해 아이템 구매 정보를 구해서 반환
    data = "No Data"

    response = {
        "result": data
    }

    return jsonify(response)

@app.route("/last-week-daily-sales", methods=['POST'])
def last_week_daily_sales():
    
    ## 최근 1주일 판매 정보 구해서 반환
    data = list()
    for idx in range(1,10):
        day_data = {"date" : (datetime.now().today() - timedelta(days=idx)).strftime("%Y-%m-%d"), "sales_amount" : idx*10000}
        data.append(day_data)

    response = {
        "result": data
    }

    return jsonify(response)

@app.route("/last-month-daily-sales", methods=['POST'])
def last_month_daily_sales():
    
    ## 최근 1달 판매 정보 구해서 반환
    data = list()
    for idx in range(1,35):
        day_data = {"date" : (datetime.now().today() - timedelta(days=idx)).strftime("%Y-%m-%d"), "sales_amount" : idx*10000}
        data.append(day_data)

    response = {
        "result": data
    }

    return jsonify(response)
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
