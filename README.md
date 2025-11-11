# Quantitative-Investing - Assistant

A free and open-source Python-based stock analysis system that combines fundamentals, 10+ technical indicators, and global macro sentiment to generate conviction-weighted BUY/SELL signals — across US, India, and Singapore markets.

> Built by Akshat Sharma, a 16-year-old student investor and quant enthusiast based in Singapore 🇸🇬.

---

## 🌍 Supported Regions

- 🇺🇸 US (NASDAQ, NYSE, S&P 500)
- 🇮🇳 India (NSE, BSE)
- 🇸🇬 Singapore (SGX, HSI)

---

## 📊 Core Features

- ✅ Multi-source OHLCV price data: Yahoo Finance, NSEpy, Stooq
- ✅ 10+ technical indicators:
  - EMA(20/50/200), RSI(14), MACD
  - Bollinger Bands, ATR, ADX, MFI, OBV, Williams %R
- ✅ Sentiment-aware overlay using:
  - ^VIX, ^GSPC, ^NDX, ^NSEI, ^STI, etc.
- ✅ Fundamentals-light:
  - P/E, ROE, Profit Margins, Debt/Equity (when available)
- ✅ Clear verdict system:
  - `STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`
- ✅ Conviction scoring & portfolio sizing:
  - Tranche-level sizing based on cash, risk profile, and allocation intent
- ✅ Live news headlines from Google News RSS and company IR feeds
- ✅ Fully offline + terminal-based — no API keys or paid sources needed

---

AAPL (Apple Inc.) | Region: US | Sector: Tech
Verdict: STRONG BUY / LONG | Conviction: 91.3%
Recommended Position: $1,370.00 (4.6% of portfolio)
Price: $195.60 | RSI: 62.4 | MACD: Positive | Sentiment: Strong
Latest News: 6 headlines (Google News + IR)

pandas
numpy
yfinance
matplotlib
feedparser
pandas_datareader
nsepy

🧠 How It Works
Loads price data from multiple sources.
Scores technical indicators.
Evaluates fundamentals when available.
Overlays broad market sentiment.
Computes composite conviction score.
Produces a clear Buy / Hold / Sell decision.

Suggests position size based on:
Portfolio size
Available cash
Risk appetite

🚧 Roadmap
Add Jupyter Notebook version
Add backtesting engine
Export to PDF report format
Build a Streamlit or web app interface

🙋‍♂️ About the Creator
Hi! I’m Akshat Sharma, a 16-year-old student investor based in Singapore.
I built this assistant to create a structured, balanced, and evidence-driven approach to stock selection and risk sizing — especially for young and beginner investors.
If you’re in finance, quant research, or trading, I’d love to connect and learn from you.

📜 License
MIT License — free to use, modify, and share.

👋 Connect With Me
LinkedIn: https://www.linkedin.com/in/akshat-sharma-255a05275/
Email: sharmaakshat2009@gmail.com
