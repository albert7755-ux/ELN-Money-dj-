import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from bs4 import BeautifulSoup
import urllib.parse
import json

# --- 1. 基礎設定 ---
st.set_page_config(page_title="結構型商品戰情室 (V13.0)", layout="wide")

# ==========================================
# 🔐 密碼保護機制
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == "5428":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("請輸入系統密碼 (Access Code)", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("請輸入系統密碼 (Access Code)", type="password", on_change=password_entered, key="password")
        st.error("❌ 密碼錯誤")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==========================================
# 🔓 主程式開始
# ==========================================

st.title("📊 結構型商品 - 關鍵點位與長週期風險回測")
st.markdown("回測區間：**2009/01/01 至今**。**特色：MoneyDJ/奇摩股市 強力抓取 (跳板模式)**。")
st.divider()

# --- 2. 側邊欄 ---
st.sidebar.header("1️⃣ 輸入標的")
default_tickers = "TSLA, NVDA, GOOG"
tickers_input = st.sidebar.text_area("股票代碼 (逗號分隔)", value=default_tickers, height=80)

st.sidebar.divider()
st.sidebar.header("2️⃣ 結構條件 (%)")
ko_pct = st.sidebar.number_input("KO (敲出價 %)", value=100.0, step=0.5)
strike_pct = st.sidebar.number_input("Strike (轉換/執行價 %)", value=80.0, step=1.0)
ki_pct = st.sidebar.number_input("KI (下檔保護價 %)", value=65.0, step=1.0)

st.sidebar.divider()
st.sidebar.header("3️⃣ 投資與配息設定")
principal = st.sidebar.number_input("投資本金 (例如 USD)", value=100000, step=10000)
coupon_pa = st.sidebar.number_input("年化配息率 (Coupon %)", value=8.0, step=0.5)

st.sidebar.divider()
st.sidebar.header("4️⃣ 回測參數設定")
period_months = st.sidebar.number_input("產品/觀察天期 (月)", min_value=1, max_value=60, value=6)

run_btn = st.sidebar.button("🚀 開始分析", type="primary")

# --- 3. 核心函數：強力爬蟲 (跳板模式) ---

@st.cache_data(ttl=3600)
def fetch_native_chinese_summary(ticker):
    """
    嘗試透過 Proxy 跳板抓取 MoneyDJ 或 Yahoo奇摩股市的原始中文資料
    """
    summary = None
    source = None

    # --- 策略 A: MoneyDJ (透過 AllOrigins 跳板) ---
    try:
        # MoneyDJ 美股個股頁面
        target_url = f"https://www.moneydj.com/us/basic/basic0001/{ticker}"
        # 使用 AllOrigins 作為跳板，繞過 IP 封鎖
        proxy_url = f"https://api.allorigins.win/get?url={urllib.parse.quote(target_url)}"
        
        response = requests.get(proxy_url, timeout=10)
        data = response.json()
        html_content = data.get('contents', '')
        
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            # MoneyDJ 的經營概述通常在特定的表格結構中，尋找關鍵字
            # 這裡用比較寬鬆的搜尋：找含有「經營概述」文字的下一個區塊
            all_text = soup.get_text(separator='\n')
            lines = all_text.split('\n')
            for i, line in enumerate(lines):
                if "經營概述" in line and len(line) < 20: # 找到標題
                    # 嘗試抓取接下來的幾行，通常是內容
                    potential_content = ""
                    for j in range(1, 10): # 往下找 10 行
                        if i+j < len(lines):
                            txt = lines[i+j].strip()
                            if len(txt) > 50: # 內容通常比較長
                                potential_content = txt
                                break
                    if potential_content:
                        summary = potential_content
                        source = "MoneyDJ 理財網 (繁體中文)"
                        break
    except Exception:
        pass # 失敗就換下一招

    # --- 策略 B: Yahoo 奇摩股市 (透過 AllOrigins 跳板) ---
    if not summary:
        try:
            # 奇摩股市美股頁面
            target_url = f"https://tw.stock.yahoo.com/quote/{ticker}/profile"
            proxy_url = f"https://api.allorigins.win/get?url={urllib.parse.quote(target_url)}"
            
            response = requests.get(proxy_url, timeout=10)
            data = response.json()
            html_content = data.get('contents', '')
            
            if html_content:
                soup = BeautifulSoup(html_content, 'html.parser')
                # 奇摩股市的簡介通常在一個 class 為 "Py(12px)" 或類似的區塊中
                # 我們找尋頁面中字數最多的段落，通常就是簡介
                paragraphs = soup.find_all('p')
                longest_p = ""
                for p in paragraphs:
                    txt = p.get_text().strip()
                    if len(txt) > len(longest_p) and len(txt) > 50:
                        longest_p = txt
                
                # 簡單過濾掉像是免責聲明之類的
                if longest_p and "報價延遲" not in longest_p:
                    summary = longest_p
                    source = "Yahoo 奇摩股市 (繁體中文)"
        except Exception:
            pass

    # --- 策略 C: 真的抓不到，回退到 yfinance 英文 (但不翻譯了，直接顯示提示) ---
    if not summary:
        try:
            tk = yf.Ticker(ticker)
            eng_summary = tk.info.get('longBusinessSummary', '')
            if eng_summary:
                summary = f"(暫無法取得中文資料，顯示原文)\n{eng_summary}"
                source = "Yahoo Finance (English)"
            else:
                summary = "查無相關公司簡介。"
                source = "系統"
        except:
            summary = "資料讀取失敗。"
            source = "系統"

    return summary, source

@st.cache_data(ttl=3600)
def get_financial_data(ticker):
    """只抓取數字數據 (EPS, PE等)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            'eps': info.get('trailingEps', 'N/A'),
            'pe': info.get('trailingPE', 'N/A'),
            'f_eps': info.get('forwardEps', 'N/A'),
            'f_pe': info.get('forwardPE', 'N/A'),
            'margin': info.get('grossMargins', 'N/A'),
            'debt': info.get('debtToEquity', 'N/A'),
        }
        # 格式化
        if isinstance(data['eps'], (int, float)): data['eps'] = f"${data['eps']:.2f}"
        if isinstance(data['pe'], (int, float)): data['pe'] = f"{data['pe']:.2f}"
        if isinstance(data['f_eps'], (int, float)): data['f_eps'] = f"${data['f_eps']:.2f}"
        if isinstance(data['f_pe'], (int, float)): data['f_pe'] = f"{data['f_pe']:.2f}"
        if isinstance(data['margin'], (int, float)): data['margin'] = f"{data['margin']*100:.2f}%"
        if isinstance(data['debt'], (int, float)): data['debt'] = f"{data['debt']:.2f}"
        return data
    except:
        return None

def display_info_card(ticker):
    """整合顯示"""
    fin_data = get_financial_data(ticker)
    desc, source = fetch_native_chinese_summary(ticker)
    
    if fin_data:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #d93025; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top:0; color:#202124;">🏢 {ticker} 企業透視</h3>
            
            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px; background: #fff; padding: 10px; border-radius: 8px;">
                <div style="flex: 1; min-width: 120px;"><b>EPS:</b> {fin_data['eps']}</div>
                <div style="flex: 1; min-width: 120px;"><b>P/E:</b> {fin_data['pe']}</div>
                <div style="flex: 1; min-width: 120px;"><b>毛利率:</b> {fin_data['margin']}</div>
                <div style="flex: 1; min-width: 120px;"><b>負債比:</b> {fin_data['debt']}</div>
            </div>

            <div style="background-color: #fff; padding: 15px; border-radius: 8px; border: 1px solid #eee;">
                <strong style="color: #d93025; font-size: 1.1em;">經營概述：</strong>
                <p style="font-size: 15px; line-height: 1.8; color: #333; text-align: justify; margin-top: 8px; margin-bottom: 0;">
                    {desc}
                </p>
                <div style="text-align: right; font-size: 12px; color: #888; margin-top: 10px;">
                    資料來源：{source} (即時抓取)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"無法取得 {ticker} 數據")

def get_stock_data_from_2009(ticker):
    try:
        start_date = "2009-01-01"
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None, f"無資料"
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        if 'Close' not in df.columns: return None, "無收盤價"
        df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA240'] = df['Close'].rolling(window=240).mean()
        return df, None
    except Exception as e: return None, str(e)

def run_backtest(df, ki_pct, strike_pct, months):
    trading_days = int(months * 21)
    bt = df[['Date', 'Close']].copy()
    bt.columns = ['Start_Date', 'Start_Price']
    bt['End_Date'] = bt['Start_Date'].shift(-trading_days)
    bt['Final_Price'] = bt['Start_Price'].shift(-trading_days)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=trading_days)
    bt['Min_Price_During'] = bt['Start_Price'].rolling(window=indexer, min_periods=1).min()
    bt = bt.dropna()
    bt['KI_Level'] = bt['Start_Price'] * (ki_pct / 100)
    bt['Strike_Level'] = bt['Start_Price'] * (strike_pct / 100)
    bt['Touched_KI'] = bt['Min_Price_During'] < bt['KI_Level']
    bt['Below_Strike'] = bt['Final_Price'] < bt['Strike_Level']
    conditions = [
        (bt['Touched_KI'] == True) & (bt['Below_Strike'] == True),
        (bt['Touched_KI'] == True) & (bt['Below_Strike'] == False),
        (bt['Touched_KI'] == False)
    ]
    bt['Result_Type'] = np.select(conditions, ['Loss', 'Safe', 'Safe'], default='Unknown')
    loss_idx = bt[bt['Result_Type'] == 'Loss'].index
    recov_days = []
    stuck = 0
    for idx in loss_idx:
        row = bt.loc[idx]
        fut = df[(df['Date'] > row['End_Date']) & (df['Close'] >= row['Strike_Level'])]
        if not fut.empty: recov_days.append((fut.iloc[0]['Date'] - row['End_Date']).days)
        else: stuck += 1
    avg_rec = np.mean(recov_days) if recov_days else 0
    total = len(bt)
    safe = (len(bt[bt['Result_Type'] == 'Safe']) / total) * 100
    pos = (len(bt[bt['Final_Price'] > bt['Start_Price']]) / total) * 100
    
    # Bar Data
    bt['Bar_Value'] = np.where(bt['Result_Type'] == 'Loss', 
                               ((bt['Final_Price'] - bt['Strike_Level'])/bt['Strike_Level'])*100, 
                               np.maximum(0, ((bt['Final_Price'] - bt['Strike_Level'])/bt['Strike_Level'])*100))
    bt['Color'] = np.where(bt['Result_Type'] == 'Loss', 'red', 'green')
    
    return bt, {'safety': safe, 'pos': pos, 'loss_cnt': len(loss_idx), 'stuck': stuck, 'rec_days': avg_rec}

def plot_chart(df, ticker, cp, ko, ki, st_p):
    plot_df = df.tail(750)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], line=dict(color='black'), name='股價'))
    fig.add_hline(y=ko, line_dash="dash", line_color="red")
    fig.add_hline(y=ki, line_dash="dot", line_color="orange")
    fig.add_hline(y=st_p, line_color="green")
    fig.update_layout(title=f"{ticker} 走勢", height=400, margin=dict(l=20,r=20,t=40,b=20))
    return fig

# --- 5. 執行 ---

if run_btn:
    ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    for ticker in ticker_list:
        # 1. 顯示中文簡介 (MoneyDJ/YahooTW 優先)
        display_info_card(ticker)
        
        # 2. 執行回測
        with st.spinner(f"計算 {ticker} 數據..."):
            df, err = get_stock_data_from_2009(ticker)
            if err:
                st.error(f"{ticker} 資料錯誤")
                continue
                
            cp = df['Close'].iloc[-1]
            p_ko = cp * (ko_pct/100)
            p_ki = cp * (ki_pct/100)
            p_st = cp * (strike_pct/100)
            
            bt_data, stats = run_backtest(df, ki_pct, strike_pct, period_months)
            
            # 配息試算 (精簡)
            m_inc = principal * (coupon_pa/100) / 12
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最新股價", f"{cp:.2f}")
            c2.metric("每月配息試算", f"${m_inc:,.0f}")
            c3.metric("本金安全率", f"{stats['safety']:.1f}%")
            
            st.plotly_chart(plot_chart(df, ticker, cp, p_ko, p_ki, p_st), use_container_width=True)
            
            # Bar Chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=bt_data['Start_Date'], y=bt_data['Bar_Value'], marker_color=bt_data['Color']))
            fig_bar.update_layout(title="歷史回測損益", height=300, margin=dict(l=20,r=20,t=40,b=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
else:
    st.info("👈 請輸入參數並開始分析。")

st.markdown("""
<style>
.disclaimer-box { background-color: #fff3f3; border: 1px solid #e0b4b4; padding: 15px; border-radius: 5px; color: #8a1f1f; font-size: 0.9em; margin-top: 30px; }
</style>
<div class='disclaimer-box'><strong>⚠️ 免責聲明</strong>：本工具僅供試算，資料來源為 MoneyDJ/Yahoo 股市 (透過 Proxy 抓取) 與 Yahoo Finance。</div>
""", unsafe_allow_html=True)
