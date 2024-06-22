

# 載入必要模組
import os
# os.chdir(r'C:\Users\user\Dropbox\系務\專題實作\112\金融看板\for students')
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 


###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">中鋼金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)


## 读取Pickle文件
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(url):
    df = pd.read_pickle(url)
    return df

df_original = load_data('2002.pkl')


# 尝试将日期列转换为 datetime 类型，处理格式不一致的情况
try:
    # 使用 format='mixed' 自动推断日期格式
    df_original['time'] = pd.to_datetime(df_original['time'], format='mixed', errors='coerce')
    st.success("日期格式转换成功!")
    
    # 检查是否有无法转换的日期
    if df_original['time'].isnull().any():
        st.warning("有部分日期无法转换为有效格式，将被移除.")
        df_original = df_original.dropna(subset=['time'])
    
except Exception as e:
    st.error(f"日期格式转换失败: {e}")

# 提供选择数据区间的界面
st.subheader("選擇開始與結束的日期, 區間:2019-01-02 至 2024-06-21")

# 获取用户输入的开始和结束日期
start_date = st.text_input('選擇開始日期 (日期格式: 2019-01-02)', '2019-01-02')
end_date = st.text_input('選擇結束日期 (日期格式: 2024-06-21)', '2024-06-21')

try:
    # 将用户输入的字符串日期转换为 datetime 对象
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # 根据用户选择的日期范围进行过滤
    df_filtered = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]
    
#except ValueError:
    # 如果日期格式不正确，显示错误信息
    #st.error("日期格式错误，请输入 'YYYY-MM-DD' 格式的日期")


###### (2) 轉化為字典 ######:
KBar_dic = df_filtered.to_dict()
#type(KBar_dic)
#KBar_dic.keys()  ## dict_keys(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'])
#KBar_dic['open']
#type(KBar_dic['open'])  ## dict
#KBar_dic['open'].values()
#type(KBar_dic['open'].values())  ## dict_values
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open']=np.array(KBar_open_list)
#type(KBar_dic['open'])  ## numpy.ndarray
#KBar_dic['open'].shape  ## (1596,)
#KBar_dic['open'].size   ##  1596

KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
#KBar_dic['product'].size   ## 1596
#KBar_dic['product'][0]      ## 'tsmc'

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
KBar_dic['time']=np.array(KBar_time_list)

# KBar_time_list[0]        ## Timestamp('2022-07-01 09:01:00')
# type(KBar_time_list[0])  ## pandas._libs.tslibs.timestamps.Timestamp
#KBar_time_list[0].to_pydatetime() ## datetime.datetime(2022, 7, 1, 9, 1)
#KBar_time_list[0].to_numpy()      ## numpy.datetime64('2022-07-01T09:01:00.000000000')
#KBar_dic['time']=np.array(KBar_time_list)
#KBar_dic['time'][80]   ## Timestamp('2022-09-01 23:02:00')

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
# Product_array = np.array([])
# Time_array = np.array([])
# Open_array = np.array([])
# High_array = np.array([])
# Low_array = np.array([])
# Close_array = np.array([])
# Volume_array = np.array([])

Date = start_date.strftime("%Y-%m-%d")

st.title("選擇 K 棒的時間長度(日 週 月 )")
options = ['日', '週', '月']
selected_option = st.selectbox('請選擇一個選項:', options)
# 根據選擇的選項自動設置 K 棒的時間長度
if selected_option == '日':
    days = st.number_input('輸入天數:', min_value=1, value=1, step=1, key="days_input")
    cycle_duration = days * 1440  # 將天數轉換為分鐘
elif selected_option == '週':
    weeks = st.number_input('輸入週數:', min_value=1, value=1, step=1, key="weeks_input")
    cycle_duration = weeks * 7 * 1440  # 將週數轉換為分鐘
elif selected_option == '月':
    months = st.number_input('輸入數:', min_value=1, value=1, step=1, key="years_input")
    cycle_duration = months * 30 * 1440  # 將年數轉換為分鐘
st.write('你選擇的時間範圍是:', selected_option)
st.write('K 棒的時間長度為:', cycle_duration, '分鐘')
#cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)',  value=default_duration, key="KBar_duration")
cycle_duration = int(cycle_duration)
#cycle_duration = 1440   ## 可以改成你想要的 KBar 週期
#KBar = indicator_f_Lo2.KBar(Date,'time',2)
KBar = indicator_forKBar_short.KBar(Date,cycle_duration)    ## 設定cycle_duration可以改成你想要的 KBar 週期



#KBar_dic['amount'].shape   ##(5585,)
#KBar_dic['amount'].size    ##5585
#KBar_dic['time'].size    ##5585

for i in range(KBar_dic['time'].size):
    
    #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
    time = KBar_dic['time'][i]
    #prod = KBar_dic['product'][i]
    open_price= KBar_dic['open'][i]
    close_price= KBar_dic['close'][i]
    low_price= KBar_dic['low'][i]
    high_price= KBar_dic['high'][i]
    qty =  KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    #tag=KBar.TimeAdd(time,price,qty,prod)
    tag=KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    # 更新K棒才判斷，若要逐筆判斷則 註解下面兩行, 因為計算 MA是利用收盤價, 而在 KBar class 中的 "TimeAdd"函數方法中, 收盤價只是一直附加最新的 price 而已.
    #if tag != 1:
        #continue
    #print(KBar.Time,KBar.GetOpen(),KBar.GetHigh(),KBar.GetLow(),KBar.GetClose(),KBar.GetVolume()) 
    
    
        
# #type(KBar.Time[1:-1]) ##numpy.ndarray       
# Time_array =  np.append(Time_array, KBar.Time[1:-1])    
# Open_array =  np.append(Open_array,KBar.Open[1:-1])
# High_array =  np.append(High_array,KBar.High[1:-1])
# Low_array =  np.append(Low_array,KBar.Low[1:-1])
# Close_array =  np.append(Close_array,KBar.Close[1:-1])
# Volume_array =  np.append(Volume_array,KBar.Volume[1:-1])
# Product_array = np.append(Product_array,KBar.Prod[1:-1])

KBar_dic = {}

# ## 形成 KBar 字典:
# KBar_dic['time'] =  Time_array   
# KBar_dic['product'] =  Product_array
# KBar_dic['open'] =  Open_array
# KBar_dic['high'] =  High_array
# KBar_dic['low'] =  Low_array
# KBar_dic['close'] =  Close_array
# KBar_dic['volume'] =  Volume_array

 ## 形成 KBar 字典 (新週期的):
KBar_dic['time'] =  KBar.TAKBar['time']   
#KBar_dic['product'] =  KBar.TAKBar['product']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] =  KBar.TAKBar['high']
KBar_dic['low'] =  KBar.TAKBar['low']
KBar_dic['close'] =  KBar.TAKBar['close']
KBar_dic['volume'] =  KBar.TAKBar['volume']
# KBar_dic['time'].shape  ## (2814,)
# KBar_dic['open'].shape  ## (2814,)
# KBar_dic['high'].shape  ## (2814,)
# KBar_dic['low'].shape  ## (2814,)
# KBar_dic['close'].shape  ## (2814,)
# KBar_dic['volume'].shape  ## (2814,)
#KBar_dic['time'][536]
######  改變 KBar 時間長度 (以上)  ########



###### (4) 計算各種技術指標 ######
##### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)


#####  (i) 移動平均線策略   #####
####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
#LongMAPeriod=st.number_input('輸入一個整數', key="Long_MA")
#LongMAPeriod=int(LongMAPeriod)
LongMAPeriod=st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
#ShortMAPeriod=st.number_input('輸入一個整數', key="Short_MA")
#ShortMAPeriod=int(ShortMAPeriod)
ShortMAPeriod=st.slider('選擇一個整數', 0, 100, 2)

#### 計算長短移動平均線
KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

valid_indices = KBar_df['MA_long'].first_valid_index()

#### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]



#####  (ii) RSI 策略   #####
#### 順勢策略
### 設定長短 RSI 的 K棒 長度:
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod=st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod=st.slider('選擇一個整數', 0, 1000, 2)

### 計算 RSI指標長短線, 以及定義中線
## 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
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

### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]


# #### 逆勢策略
# ### 建立部位管理物件
# OrderRecord=Record() 
# ### 計算 RSI指標, 天花板與地板
# RSIPeriod=5
# Ceil=80
# Floor=20
# MoveStopLoss=30
# KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
# KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
# KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))

# ### 將K線 Dictionary 轉換成 Dataframe
# KBar_RSI_df=pd.DataFrame(KBar_dic)

# MACD策略
#設定macd k棒長度
st.subheader("設定計算MACD的週期")
ShortEMA = st.slider('短期EMA (例如12)', 1, 100, 12)
LongEMA = st.slider('長期EMA (例如26)', 1, 100, 26)
SignalEMA = st.slider('信號EMA (例如9)', 1, 100, 9)

##計算macd

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    delta = df['close'].diff()
    # 計算短期EMA
    short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
    # 計算長期EMA
    long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
    # 計算MACD線
    macd_line = short_ema - long_ema
    # 計算信號線
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    # 計算MACD柱
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

# 將 MACD 計算結果加入 DataFrame
KBar_df['MACD_line'], KBar_df['Signal_line'], KBar_df['MACD_histogram'] = calculate_macd(KBar_df)

KBar_df['EMA_short'] = KBar_df['close'].ewm(span=ShortEMA, adjust=False).mean()
KBar_df['EMA_long'] = KBar_df['close'].ewm(span=LongEMA, adjust=False).mean()
KBar_df['MACD'] = KBar_df['EMA_short'] - KBar_df['EMA_long']
KBar_df['Signal'] = KBar_df['MACD'].ewm(span=SignalEMA, adjust=False).mean()
KBar_df['Hist'] = KBar_df['MACD'] - KBar_df['Signal']

#計算布林通道
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


# KBar_df = pd.DataFrame(KBar_dic)
KBar_df['MA'] = KBar_df['close'].rolling(window=BollingerPeriod).mean()
KBar_df['STD'] = KBar_df['close'].rolling(window=BollingerPeriod).std()
KBar_df['Upper'] = KBar_df['MA'] + (KBar_df['STD'] * BollingerStdDev)
KBar_df['Lower'] = KBar_df['MA'] - (KBar_df['STD'] * BollingerStdDev)

# 選擇唐奇安通道周期
st.subheader("設定唐奇安通道的周期（天數）")
DonchianPeriod = st.slider('選擇唐奇安通道的天數', 1, 100, 20)

# 計算唐奇安通道
def calculate_donchian_channel(df, period=20):
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    delta = df['high'].diff()
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    middle_band = (high_max + low_min) / 2
    return high_max, middle_band, low_min

KBar_df['Donchian_upper'], KBar_df['Donchian_middle'], KBar_df['Donchian_lower'] = calculate_donchian_channel(KBar_df, DonchianPeriod)


###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]


###### (6) 畫圖 ######
st.subheader("畫圖")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
import plotly.offline as pyoff


##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)

##### K線圖, RSI
with st.expander("K線圖, 長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    #### include candlestick with rangeselector
    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'),
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'),
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
    st.plotly_chart(fig2, use_container_width=True)

##### K線圖, MACD
with st.expander("K線圖, MACD"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    #fig3.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MACD'], mode='lines', line=dict(color='blue', width=2), name='MACD'), secondary_y=True)
    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Signal'], mode='lines', line=dict(color='red', width=2), name='信號線'), secondary_y=True)
    fig3.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Hist'], name='MACD 柱狀圖', marker=dict(color='green')), secondary_y=True)
    fig3.layout.yaxis2.showgrid = True
    st.plotly_chart(fig3, use_container_width=True)

##### K線圖, 布林通道
with st.expander("K線圖, 布林通道"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    #fig4.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MA'], mode='lines', line=dict(color='blue', width=2), name='移動平均線'), secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Upper'], mode='lines', line=dict(color='green', width=2), name='上軌'), secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Lower'], mode='lines', line=dict(color='red', width=2), name='下軌'), secondary_y=True)
    fig4.layout.yaxis2.showgrid = True
    st.plotly_chart(fig4, use_container_width=True)
    
    # 繪製K線圖與唐奇安通道
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
