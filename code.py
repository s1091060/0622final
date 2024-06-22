import os
import numpy as np
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">中鋼金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(url):
    df = pd.read_pickle(url)
    return df

df_original = load_data('2002.pkl')

try:
    df_original['time'] = pd.to_datetime(df_original['time'], format='mixed', errors='coerce')
    st.success("日期格式轉換成功")
    if df_original['time'].isnull().any():
        st.warning("有部分日期無法轉換成有效格式，將被移除.")
        df_original = df_original.dropna(subset=['time'])
    
except Exception as e:
    st.error(f"日期格式轉換失敗: {e}")

st.subheader("選擇開始與結束的日期, 區間:2019-01-02 至 2024-06-21")
start_date = st.text_input('選擇開始日期 (日期格式: 2019-01-02)', '2019-01-02')
end_date = st.text_input('選擇結束日期 (日期格式: 2024-06-21)', '2024-06-21')

try:
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    df_filtered = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]
    
#except ValueError: 
    #st.error("日期格式錯誤，入 'YYYY-MM-DD' 格式的日期")

###### (2) 轉化為字典 ######:
KBar_dic = df_filtered.to_dict()

KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open']=np.array(KBar_open_list)
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
KBar_dic['time']=np.array(KBar_time_list)

KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low']=np.array(KBar_low_list)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high']=np.array(KBar_high_list)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close']=np.array(KBar_close_list)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume']=np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount']=np.array(KBar_amount_list)


######  (3) 改變 KBar 時間長度 (以下)  ########
Date = start_date.strftime("%Y-%m-%d")

st.title("選擇 K 棒的時間長度(日 週 月 )")
options = ['日', '週', '月']
selected_option = st.selectbox('請選擇一個選項:', options)

if selected_option == '日':
    days = st.number_input('輸入天數:', min_value=1, value=1, step=1, key="days_input")
    cycle_duration = days * 1440  # 將天數轉換為分鐘
elif selected_option == '週':
    weeks = st.number_input('輸入週數:', min_value=1, value=1, step=1, key="weeks_input")
    cycle_duration = weeks * 7 * 1440  # 將週數轉換為分鐘
elif selected_option == '月':
    months = st.number_input('輸入數:', min_value=1, value=1, step=1, key="years_input")
    cycle_duration = months * 30 * 1440  # 將月數轉換為分鐘
st.write('你選擇的時間範圍是:', selected_option)
st.write('K 棒的時間長度為:', cycle_duration, '分鐘')
cycle_duration = int(cycle_duration)
KBar = indicator_forKBar_short.KBar(Date,cycle_duration)   

for i in range(KBar_dic['time'].size):
    
    time = KBar_dic['time'][i]
    open_price= KBar_dic['open'][i]
    close_price= KBar_dic['close'][i]
    low_price= KBar_dic['low'][i]
    high_price= KBar_dic['high'][i]
    qty =  KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    tag=KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
KBar_dic = {}

KBar_dic['time'] =  KBar.TAKBar['time']   
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] =  KBar.TAKBar['high']
KBar_dic['low'] =  KBar.TAKBar['low']
KBar_dic['close'] =  KBar.TAKBar['close']
KBar_dic['volume'] =  KBar.TAKBar['volume']

###### (4) 計算各種技術指標 ######
KBar_df = pd.DataFrame(KBar_dic)

st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
LongMAPeriod=st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
ShortMAPeriod=st.slider('選擇一個整數', 0, 100, 2)

KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

valid_indices = KBar_df['MA_long'].first_valid_index()

last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod=st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod=st.slider('選擇一個整數', 0, 1000, 2)

def calculate_rsi(df, period=14):
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle']=np.array([50]*len(KBar_dic['time']))

last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

st.subheader("設定計算MACD的週期")
ShortEMA = st.slider('短期EMA (例如12)', 1, 100, 12)
LongEMA = st.slider('長期EMA (例如26)', 1, 100, 26)
SignalEMA = st.slider('信號EMA (例如9)', 1, 100, 9)

##計算macd

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    delta = df['close'].diff()
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

KBar_df['MACD_line'], KBar_df['Signal_line'], KBar_df['MACD_histogram'] = calculate_macd(KBar_df)

KBar_df['EMA_short'] = KBar_df['close'].ewm(span=ShortEMA, adjust=False).mean()
KBar_df['EMA_long'] = KBar_df['close'].ewm(span=LongEMA, adjust=False).mean()
KBar_df['MACD'] = KBar_df['EMA_short'] - KBar_df['EMA_long']
KBar_df['Signal'] = KBar_df['MACD'].ewm(span=SignalEMA, adjust=False).mean()
KBar_df['Hist'] = KBar_df['MACD'] - KBar_df['Signal']

st.subheader("設定計算布林通道的週期和標準差倍數")
BollingerPeriod = st.slider('選擇布林通道週期 (例如20)', 1, 100, 20)
BollingerStdDev = st.slider('選擇標準差倍數 (例如2)', 1, 5, 2)

def calculate_bollinger_bands(df, period=20, std_dev_multiplier=2):
    delta = df['close'].diff()
    middle_band = df['close'].rolling(window=period).mean()
    standard_deviation = df['close'].rolling(window=period).std()
    upper_band = middle_band + (standard_deviation * std_dev_multiplier)
    lower_band = middle_band - (standard_deviation * std_dev_multiplier)
    return  middle_band, upper_band, lower_band

KBar_df['MA'] = KBar_df['close'].rolling(window=BollingerPeriod).mean()
KBar_df['STD'] = KBar_df['close'].rolling(window=BollingerPeriod).std()
KBar_df['Upper'] = KBar_df['MA'] + (KBar_df['STD'] * BollingerStdDev)
KBar_df['Lower'] = KBar_df['MA'] - (KBar_df['STD'] * BollingerStdDev)

st.subheader("設定唐奇安通道的周期（天數）")
DonchianPeriod = st.slider('選擇唐奇安通道的天數', 1, 100, 20)

def calculate_donchian_channel(df, period=20):
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    delta = df['high'].diff()
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    middle_band = (high_max + low_min) / 2
    return high_max, middle_band, low_min

KBar_df['Donchian_upper'], KBar_df['Donchian_middle'], KBar_df['Donchian_lower'] = calculate_donchian_channel(KBar_df, DonchianPeriod)

KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]

st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.offline as pyoff

with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)  
    
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)

with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)  
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'),
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'),
                  secondary_y=False)
    fig2.layout.yaxis2.showgrid=True
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("K線圖, MACD"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MACD'], mode='lines', line=dict(color='blue', width=2), name='MACD'), secondary_y=True)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Signal'], mode='lines', line=dict(color='red', width=2), name='信號線'), secondary_y=True)
    fig3.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Hist'], name='MACD 柱狀圖', marker=dict(color='green')), secondary_y=True)
    fig3.layout.yaxis2.showgrid = True
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("K線圖, 布林通道"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MA'], mode='lines', line=dict(color='blue', width=2), name='移動平均線'), secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Upper'], mode='lines', line=dict(color='green', width=2), name='上軌'), secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Lower'], mode='lines', line=dict(color='red', width=2), name='下軌'), secondary_y=True)
    fig4.layout.yaxis2.showgrid = True
    st.plotly_chart(fig4, use_container_width=True)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with st.expander("K線圖, 唐奇安通道圖"):
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    fig5.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig5.add_trace(go.Scatter(x=KBar_df['Time'][DonchianPeriod-1:], y=KBar_df['Donchian_upper'][DonchianPeriod-1:], name='唐奇安上軌', mode='lines', line=dict(color='blue')))
    fig5.add_trace(go.Scatter(x=KBar_df['Time'][DonchianPeriod-1:], y=KBar_df['Donchian_middle'][DonchianPeriod-1:], name='唐奇安中軌', mode='lines', line=dict(color='green')))
    fig5.add_trace(go.Scatter(x=KBar_df['Time'][DonchianPeriod-1:], y=KBar_df['Donchian_lower'][DonchianPeriod-1:], name='唐奇安下軌', mode='lines', line=dict(color='red')))
    st.plotly_chart(fig5, use_container_width=True)
