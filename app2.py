from flask import Flask, jsonify, render_template
from flask_cors import CORS
from threading import Thread
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import talib
import datetime

app = Flask(__name__)
CORS(app)

symbol = "EURUSD"
volume = 0.5
TIMEFRAME = mt5.TIMEFRAME_M5
SMA_PERIOD = 10
DEVIATION = 20
count = 100
historical_data_points = 100

def initialize_mt5():
    mt5.initialize()

def get_ohlc_data_with_volume(symbol, timeframe, count, include_current=True):
    try:
        if include_current:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)

        ohlc_data = [{'timestamp': x['time'], 'open': x['open'], 'high': x['high'], 'low': x['low'],
                      'close': x['close'], 'volume': x['tick_volume']} for x in rates]
        return ohlc_data
    except Exception as e:
        print(f"An error occurred while retrieving OHLC data: {e}")
        return None
 
# Function to send a market order
def market_order(symbol, volume, order_type, deviation=DEVIATION, stop_loss=None, take_profit=None):
 
    if order_type not in ['buy', 'sell']:
        raise ValueError("Invalid order type. It should be 'buy' or 'sell'.")
 
    tick = mt5.symbol_info_tick(symbol)
 
    order_type_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}
 
    if order_type == 'buy':
        price = price_dict['buy']
 
        tp = price + 30 * mt5.symbol_info(symbol).point
    else:
        price = price_dict['sell']
        tp = price - 30 * mt5.symbol_info(symbol).point
 
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_dict[order_type],
        "price": price_dict[order_type],
        "deviation": deviation,
        "magic": 100,
        "comment": f"Python {order_type} order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
 
    if take_profit is not None:
        request["tp"] = take_profit
    else:
        request["tp"] = tp  # Set the calculated take profit
 
    # Send the order request
    result = mt5.order_send(request)
 
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to execute order: {result.comment}")
    else:
        print(f"Order placed successfully: {result.order}")
 
    # Store the order details for profit/loss calculation
    order_details = {
        "symbol": symbol,
        "volume": volume,
        "type": order_type_dict[order_type],
        "open_price": price_dict[order_type],  # Store the open price
    }
 
    return result
 
# Function to close an order based on ticket id
def close_order(ticket, deviation=DEVIATION):
    positions = mt5.positions_get()
 
    for pos in positions:
        if pos.ticket == ticket:
            tick = mt5.symbol_info_tick(pos.symbol)
            price_dict = {0: tick.ask, 1: tick.bid}
            type_dict = {0: 1, 1: 0}  # 0 represents buy, 1 represents sell - inverting order_type to close the position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],  # Close the position at the current market price
                "deviation": deviation,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
 
            print(f"Closing order {pos.ticket} for {pos.volume} lots of {pos.symbol}...")
 
            order_result = mt5.order_send(request)
            print(order_result)
 
            return order_result
 
    print(f"Ticket {ticket} does not exist")
    return 'Ticket does not exist'
 
 
def get_ohlc_data_with_volume(symbol, timeframe, count, include_current=True):
    try:
        if include_current:
            # Fetch the latest data along with the historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 1)
        else:
            # Fetch only historical data
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)
 
        ohlc_data = [{'timestamp': x['time'], 'open': x['open'], 'high': x['high'], 'low': x['low'], 'close': x['close'], 'volume': x['tick_volume']} for x in rates]
        return ohlc_data
    except Exception as e:
        print(f"An error occurred while retrieving OHLC data: {e}")
        return None
 
data = get_ohlc_data_with_volume(symbol, TIMEFRAME, count)
data = pd.DataFrame(data)
def get_exposure(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        pos_df = pd.DataFrame(positions, columns=positions[0]._asdict().keys())
        exposure = pos_df['volume'].sum()
        return exposure
    else:
        return 0  # or return another appropriate value
def klinger_volume_oscillator(data, symbol, fast_length=35, slow_length=50, signal_smoothing_type="EMA", signal_smoothing_length=16):
    # Fetch volume from the data
    volume = data["volume"]
 
    # Check if volume is None
    if volume is None:
        print("Unable to retrieve volume data.")
        return pd.Series(), pd.Series(), pd.Series()
 
    mom = data["close"].diff()
 
    trend = np.zeros(len(data))
    trend[0] = 0.0
 
    for i in range(1, len(data)):
        if np.isnan(trend[i - 1]):
            trend[i] = 0
        else:
            if mom[i] > 0:
                trend[i] = 1
            elif mom[i] < 0:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]
 
    dm = data["high"] - data["low"]
    cm = np.zeros(len(data))
    cm[0] = 0.0
 
    for i in range(1, len(data)):
        if np.isnan(cm[i - 1]):
            cm[i] = 0.0
        else:
            if trend[i] == trend[i - 1]:
                cm[i] = cm[i - 1] + dm[i]
            else:
                cm[i] = dm[i] + dm[i - 1]
 
    vf = np.zeros(len(data))
    for i in range(len(data)):
        if cm[i] != 0:
            vf[i] = 100 * volume[i] * trend[i] * abs(2 * dm[i] / cm[i] - 1)
 
    # Convert fast_length and slow_length to integers
    fast_length = int(fast_length)
    slow_length = int(slow_length)
 
    kvo = pd.Series(vf).ewm(span=fast_length).mean() - pd.Series(vf).ewm(span=slow_length).mean()
 
    if signal_smoothing_type == "EMA":
        signal = kvo.ewm(span=signal_smoothing_length).mean()
    else:
        signal = kvo.rolling(window=signal_smoothing_length).mean()
 
    hist = kvo - signal
 
    return kvo, signal, hist
 
# Function to calculate Smooth Average Range (smrng)
def smoothrng(x, t, m):
    wper = t * 2 - 1
    avrng = abs(x - x.shift(1)).ewm(span=t, adjust=False).mean()
    smoothrng = avrng.ewm(span=wper, adjust=False).mean() * m
    return smoothrng
 
# Function to calculate Range Filter (filt)
def rngfilt(x, r):
    rngfilt = x.copy()
    for i in range(1, len(x)):
        if x[i] > rngfilt[i - 1]:
            rngfilt[i] = x[i] - r if x[i] - r > rngfilt[i - 1] else rngfilt[i - 1]
        else:
            rngfilt[i] = x[i] + r if x[i] + r < rngfilt[i - 1] else rngfilt[i - 1]
    return rngfilt
 
# Function to calculate Range Filter Direction (upward and downward)
def filter_direction(x):
    upward = np.zeros(len(x))
    downward = np.zeros(len(x))
 
    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            upward[i] = upward[i - 1] + 1
        elif x[i] < x[i - 1]:
            downward[i] = downward[i - 1] + 1
 
    return upward, downward
 
# Function to calculate Target Bands (hband and lband)
def target_bands(x, smrng):
    hband = x + smrng
    lband = x - smrng
    return hband, lband
 
# Function to calculate Break Outs (longCond and shortCond)
def break_outs(src, filt, upward, downward):
    longCond = np.zeros(len(src))
    shortCond = np.zeros(len(src))
 
    for i in range(1, len(src)):
        if src[i] > filt[i] and src[i] > src[i - 1] and upward[i] > 0:
            longCond[i] = 1
        elif src[i] > filt[i] and src[i] < src[i - 1] and upward[i] > 0:
            longCond[i] = 1
        elif src[i] < filt[i] and src[i] < src[i - 1] and downward[i] > 0:
            shortCond[i] = 1
        elif src[i] < filt[i] and src[i] > src[i - 1] and downward[i] > 0:
            shortCond[i] = 1
 
    CondIni = np.zeros(len(src))
    longCondition = np.zeros(len(src))
    shortCondition = np.zeros(len(src))
 
    for i in range(1, len(src)):
        if longCond[i]:
            CondIni[i] = 1
        elif shortCond[i]:
            CondIni[i] = -1
 
    for i in range(1, len(src)):
        if longCond[i] and CondIni[i - 1] == -1:
            longCondition[i] = 1
        elif shortCond[i] and CondIni[i - 1] == 1:
            shortCondition[i] = 1
 
    return longCondition, shortCondition
 
# Function to calculate Average True Range (ATR)
def calculate_atr(data, periods=14):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=periods, min_periods=1).mean()
 
    return atr
 
# Function to calculate Super Trend
def super_trend(data, atr_multiplier, periods):
    atr = calculate_atr(data, periods)
    hl2 = (data['high'] + data['low']) / 2
    up = hl2 - (atr_multiplier * atr)
    dn = hl2 + (atr_multiplier * atr)
    trend = pd.Series(np.where(data['close'] > up.shift(), 1, np.where(data['close'] < dn.shift(), -1, 0)))
 
    for i in range(1, len(trend)):
        if trend.iloc[i] == -1 and trend.iloc[i - 1] == -1:
            dn.iloc[i] = max(dn.iloc[i], dn.iloc[i - 1])
        elif trend.iloc[i] == 1 and trend.iloc[i - 1] == 1:
            up.iloc[i] = min(up.iloc[i], up.iloc[i - 1])
 
    super_trend = pd.Series(np.where(trend == 1, up, dn))
    return super_trend, trend
 

 
    previous_trend = None
stop_trading = False

def start_trading():
    global stop_trading

    while not stop_trading:
        ohlc_data = get_ohlc_data_with_volume(symbol, TIMEFRAME, count, include_current=True)
        data = pd.DataFrame(ohlc_data)

        # Other calculations
        super_trend_data, trend = super_trend(data, 1.0, SMA_PERIOD)
        last_trend = trend.iloc[-1]
        super_trend_value = super_trend_data.iloc[-1]

        kvo, signal, hist = klinger_volume_oscillator(data, symbol)
        current_hist_color = 'Unknown'

        if len(hist) > 0:
            current_hist_color = 'Green' if hist.iloc[-1] > 0 else "Red"

        kvo_signal = 'BUY' if abs(kvo.iloc[-1]) > abs(signal.iloc[-1]) else 'SELL'
        kvo_strength = 'Strong' if abs(kvo.iloc[-1]) > abs(signal.iloc[-1]) else 'Weak'

        # Execute trade based on signals
        if current_hist_color == 'Green' and kvo_signal == 'BUY':
            # Close all short positions
            for pos in mt5.positions_get():
                if pos.type == 1:
                    print(f"Closing short position {pos.ticket}...")
                    close_order(pos.ticket)

            # Open a new short position with TP and SL
            if not mt5.positions_total():
                print("Placing a BUY order...")
                market_order(symbol, volume, 'buy')

        elif current_hist_color == 'Red' and kvo_signal =='SELL':
            # Close all long positions
            for pos in mt5.positions_get():
                if pos.type == 0:
                    print(f"Closing long position {pos.ticket}...")
                    close_order(pos.ticket)

            # Open a new long position with TP and SL
            if not mt5.positions_total():
                print("Placing a SELL order...")
                market_order(symbol, volume, 'sell')

        # Print the trading signals and conditions
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{current_time}\n"
              f"Super Trend: {super_trend_value}\n"
              f"Current Trend: {last_trend}\n"
              f"KVO Signal Confirmed: {kvo_signal}\n"
              f"KVO Histo: {current_hist_color}\n"
              f"KVO Strength: {kvo_strength}\n"
              f"--------------------------")

        time.sleep(30)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_trading')
def start_trading_route():
    global trading_thread
    if not trading_thread.is_alive():
        trading_thread = Thread(target=start_trading)
        trading_thread.start()
        return jsonify({'message': 'Trading bot has been started successfully'})
    else:
        return jsonify({'message': 'Trading is already running!'})

@app.route('/stop_trading')
def stop_trading_route():
    global stop_trading
    stop_trading = True
    return jsonify({'message': 'Trading bot has been stopped successfully'})

if __name__ == '__main__':
    initialize_mt5()
    trading_thread = Thread(target=start_trading)
    app.run(debug=True)
 