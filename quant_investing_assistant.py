#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quant Assistant v6 — Global + India + Singapore + Live News (Free Sources)
- Multi-provider OHLCV (Yahoo, NSEpy, Stooq) with timezone-normalized indices
- 10+ technical indicators (EMA cross, RSI, MACD, ADX, ATR, OBV, MFI, Bollinger, W%R)
- Region-aware sentiment (^GSPC/^NDX/^VIX or ^NSEI/^NSEBANK/^INDIAVIX or ^STI/^HSI/^VIX)
- Fundamentals-lite (PE/ROE/margins/debt if available)
- Strong, clear verdicts: STRONG BUY/LONG, BUY, HOLD, SELL, STRONG SELL/SHORT
- Conviction & risk sizing (risk caps; tranche sizing)
- Defaults for all inputs (Economy: US, Cash: 1000, Desired: 5%, Stock: MSFT, Asset: Stocks)
- Live news from free RSS (Google News + known IR feeds) — no API keys
"""

import os, math, sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# Optional providers
try:
    from nsepy import get_history as nse_get_history
    HAVE_NSEPY = True
except Exception:
    HAVE_NSEPY = False

try:
    from pandas_datareader import data as pdr
    HAVE_PDR = True
except Exception:
    HAVE_PDR = False

# Lightweight RSS news (no API key)
try:
    import feedparser
    HAVE_FEEDPARSER = True
except Exception:
    HAVE_FEEDPARSER = False


# ================= Utility =================
def fmt(x, d=2):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{x:.{d}f}"
    except Exception:
        return "—"

def pct(x):
    return f"{x*100:.1f}%" if x is not None and not (isinstance(x, float) and math.isnan(x)) else "—"

def clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _ensure_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    else:
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = out[~out.index.isna()].sort_index()
    return out


# ================= Data layer (multi-provider, free) =================
def _std_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    # Normalize column names
    def pick(name):
        for c in out.columns:
            if c.lower().replace(" ", "") == name.lower().replace(" ", ""):
                return c
        return None

    ren = {}
    for want in ["Open","High","Low","Close","Adj Close","Volume"]:
        c = pick(want)
        if c and c != want:
            ren[c] = want
    out = out.rename(columns=ren)

    for need in ["Open","High","Low","Close","Adj Close","Volume"]:
        if need not in out.columns:
            out[need] = np.nan

    out = out[["Open","High","Low","Close","Adj Close","Volume"]]
    out = _ensure_utc_naive_index(out)  # ✅ normalize index (tz-naive UTC)
    return out

def yf_history(symbol: str, years=5, interval="1d"):
    try:
        period = f"{max(1,int(years))}y"
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        return _std_df(df)
    except Exception:
        return pd.DataFrame()

def nse_history(symbol_no_suffix: str, years=5):
    if not HAVE_NSEPY: return pd.DataFrame()
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=365*max(1,int(years)))
        df = nse_get_history(symbol=symbol_no_suffix, start=start, end=end)
        if df is None or df.empty: return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return _std_df(df)
    except Exception:
        return pd.DataFrame()

def stooq_history(symbol: str, years=5):
    if not HAVE_PDR: return pd.DataFrame()
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=365*max(1,int(years)))
        df = pdr.DataReader(symbol, "stooq", start, end)
        return _std_df(df)
    except Exception:
        return pd.DataFrame()

def normalize_symbol(asset_class: str, region: str, raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s: return s
    if asset_class in {"Stocks","Funds/ETFs"}:
        if region == "India" and not (s.endswith(".NS") or s.endswith(".BO")):
            s = s + ".NS"
        elif region == "Singapore" and not s.endswith(".SI"):
            s = s + ".SI"
    return s

def history_multi(asset_class: str, region: str, symbol: str, years=5):
    used = []
    df = pd.DataFrame()

    if region == "India":
        base = symbol.replace(".NS","").replace(".BO","")
        sym = symbol if (symbol.endswith(".NS") or symbol.endswith(".BO")) else f"{symbol}.NS"
        providers = [
            ("NSEpy", lambda: nse_history(base, years)),
            ("Yahoo", lambda: yf_history(sym, years)),
            ("Stooq", lambda: stooq_history(sym, years)),
        ]
    elif region == "Singapore":
        sym = symbol if symbol.endswith(".SI") else f"{symbol}.SI"
        providers = [
            ("Yahoo", lambda: yf_history(sym, years)),
            ("Stooq", lambda: stooq_history(sym, years)),
        ]
    else:
        providers = [
            ("Yahoo", lambda: yf_history(symbol, years)),
            ("Stooq", lambda: stooq_history(symbol, years)),
        ]

    for name, fn in providers:
        d = fn()
        d = _ensure_utc_naive_index(d)
        if d is not None and not d.empty:
            used.append(name)
            if df.empty:
                df = d
            else:
                # combine_first requires consistent index types — already normalized above
                df = df.combine_first(d)
    return df, used


# ================= Indicators (10+) =================
def sma(s,n): return s.rolling(n).mean()
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100/(1+rs))
def macd(s, f=12, sl=26, sig=9):
    m = ema(s,f) - ema(s,sl)
    return m, ema(m,sig)
def bollinger(s,n=20):
    mid = sma(s,n); sd = s.rolling(n).std()
    return mid, mid+2*sd, mid-2*sd
def atr(h,l,c,n=14):
    tr = pd.concat([(h-l).abs(), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def obv(c,v): return (np.sign(c.diff())*v).fillna(0).cumsum()
def mfi(h,l,c,v,n=14):
    tp=(h+l+c)/3; pmf=tp.where(tp>tp.shift(),0)*v; nmf=tp.where(tp<tp.shift(),0)*v
    mr = pmf.rolling(n).sum() / (nmf.rolling(n).sum()+1e-9)
    return 100 - (100/(1+mr))
def adx(h,l,c,n=14):
    plus_dm = h.diff(); minus_dm = l.diff().abs()
    plus_dm[plus_dm<0]=0; minus_dm[minus_dm<0]=0
    tr = atr(h,l,c,n)
    plus_di = 100*(plus_dm.rolling(n).sum()/(tr+1e-9))
    minus_di= 100*(minus_dm.rolling(n).sum()/(tr+1e-9))
    dx = (abs(plus_di-minus_di)/((plus_di+minus_di)+1e-9))*100
    return dx.rolling(n).mean()
def williams_r(h,l,c,n=14):
    hh=h.rolling(n).max(); ll=l.rolling(n).min()
    return -100*(hh-c)/(hh-ll+1e-9)


# ================= Technical scoring =================
def score_technicals(df: pd.DataFrame, holding_months: int):
    c,h,l,v = df["Close"], df["High"], df["Low"], df["Volume"]
    short,long_ = (20,50) if holding_months<=3 else (50,200)
    ema_s, ema_l = ema(c, short), ema(c, long_)
    r14 = rsi(c,14); macd_line, macd_sig = macd(c); adx14 = adx(h,l,c,14)
    mfi14 = mfi(h,l,c,v,14); atr14 = atr(h,l,c,14); obv_val = obv(c,v)
    bb_mid, bb_up, bb_lo = bollinger(c,20); wr14 = williams_r(h,l,c,14)

    s=0.0
    s += 0.15 if ema_s.iloc[-1] > ema_l.iloc[-1] else -0.15
    s += 0.10 if macd_line.iloc[-1] > macd_sig.iloc[-1] else -0.10
    s += 0.10 if 45 < r14.iloc[-1] < 70 else (-0.10 if r14.iloc[-1] > 75 or r14.iloc[-1] < 30 else 0)
    s += 0.06 if adx14.iloc[-1] > 25 else -0.04
    s += 0.04 if mfi14.iloc[-1] > 50 else -0.04
    s += 0.04 if wr14.iloc[-1] > -60 else (-0.02 if wr14.iloc[-1] < -80 else 0)
    s += 0.04 if obv_val.iloc[-1] > obv_val.iloc[-10] else -0.04
    s += 0.03 if (bb_mid.iloc[-1] < c.iloc[-1] < bb_up.iloc[-1]) else (-0.03 if c.iloc[-1] < bb_mid.iloc[-1] else 0)
    vol_ratio = float((atr14.iloc[-1]/c.iloc[-1])) if c.iloc[-1] else 0.0
    s += 0.04 if vol_ratio < 0.04 else -0.04

    score = clamp01((s+1)/2.0)
    snapshot = dict(
        price=float(c.iloc[-1]), ema_short=float(ema_s.iloc[-1]), ema_long=float(ema_l.iloc[-1]),
        rsi=float(r14.iloc[-1]), macd=float(macd_line.iloc[-1]), macd_signal=float(macd_sig.iloc[-1]),
        adx=float(adx14.iloc[-1]), mfi=float(mfi14.iloc[-1]), wr=float(wr14.iloc[-1]),
        atr=float(atr14.iloc[-1]), obv=float(obv_val.iloc[-1]),
        bb_mid=float(bb_mid.iloc[-1]), bb_up=float(bb_up.iloc[-1]), bb_lo=float(bb_lo.iloc[-1])
    )
    return score, snapshot


# ================= Fundamentals (lightweight) =================
def fundamentals_score(info: dict):
    if not info: return 0.6
    s=0.5
    pe_t = info.get("trailingPE"); pe_f = info.get("forwardPE")
    roe = info.get("returnOnEquity"); d2e = info.get("debtToEquity"); pm = info.get("profitMargins")
    if pe_t and pe_t<20: s += 0.07
    if pe_f and pe_f<22: s += 0.07
    if roe and roe>0.15: s += 0.10
    if d2e and d2e<100: s += 0.05
    if pm and pm>0.10: s += 0.08
    return clamp01(s)


# ================= Sentiment (region-aware) =================
def market_sentiment(region: str):
    if region == "India":
        idx1, idx2, vol = "^NSEI","^NSEBANK","^INDIAVIX"
    elif region == "Singapore":
        idx1, idx2, vol = "^STI","^HSI","^VIX"
    else:
        idx1, idx2, vol = "^GSPC","^NDX","^VIX"
    packs = {k:yf.Ticker(k).history(period="6mo") for k in [idx1,idx2,vol]}
    def ret(df,n=21):
        try:
            return float(df["Close"].iloc[-1]/df["Close"].iloc[-n]-1) if len(df)>n else 0.0
        except Exception:
            return 0.0
    r1m = ret(packs[idx1],21); r3m = ret(packs[idx2],63)
    vnow = float(packs[vol]["Close"].iloc[-1]) if not packs[vol].empty else 20.0
    def nret(x): return 0.5 + x/0.20  # -10%->0, +10%->1 (clamped later)
    vlow, vhigh = (12, 28) if region!="US/Global" else (14, 30)
    if vnow <= vlow: vscore = 1.0
    elif vnow >= vhigh: vscore = 0.2
    else: vscore = 1.0 - (vnow - vlow) * (0.8/(vhigh-vlow))
    score = clamp01(0.65*clamp01((nret(r1m)+nret(r3m))/2.0) + 0.35*vscore)
    breakdown = dict(idx1_1m=r1m, idx2_3m=r3m, vol=vnow, indices=(idx1,idx2), vol_symbol=vol)
    return score, breakdown


# ================= Verdict & Sizing =================
def verdict(fscore, tscore, sentiment, holding_months):
    if holding_months <= 3: wf, wt = 0.45, 0.55
    elif holding_months <= 6: wf, wt = 0.55, 0.45
    elif holding_months <= 12: wf, wt = 0.62, 0.38
    else: wf, wt = 0.70, 0.30
    base = clamp01(wf*fscore + wt*tscore + (sentiment-0.5)*0.06)
    if base >= 0.82 and fscore >= 0.70: return base, "STRONG BUY / LONG"
    if base >= 0.68: return base, "BUY"
    if base >= 0.52: return base, "HOLD / NEUTRAL"
    if base >= 0.38: return base, "SELL / REDUCE"
    return base, "STRONG SELL / SHORT"

def risk_cap_pct(risk_level):
    r = (risk_level or "medium").lower()
    return 2.0 if r=="low" else (12.0 if r=="high" else 5.0)  # percent of portfolio

def allocation_recommendation(conviction_pct, risk_level, desired_pct):
    cap = risk_cap_pct(risk_level)
    target = min(max(desired_pct, 0.0), cap)
    return round(target * (conviction_pct/100.0), 2), cap


# ================= Explanations =================
def explain_choice(name, verdict_text, t, sent, info):
    bullets = []
    if "BUY" in verdict_text:
        bullets.append("Momentum is improving (MACD > Signal) while trend stays constructive (ADX > 25).")
        if t.get("rsi") and t["rsi"] < 70:
            bullets.append("RSI is not overbought, giving room for upside if it reclaims 50–60.")
    if "SELL" in verdict_text:
        bullets.append("Momentum deterioration (RSI < 50 or bearish MACD cross) warrants caution.")
    if sent >= 0.6:
        bullets.append("Broader market sentiment is supportive, a tailwind for entries.")
    if sent < 0.4:
        bullets.append("Market risk appetite is soft; prefer staggered sizing and tighter stops.")
    if info:
        if info.get("profitMargins", 0) and info["profitMargins"] > 0.1:
            bullets.append("Healthy profitability profile supports medium-term compounding.")
        if info.get("returnOnEquity", 0) and info["returnOnEquity"] > 0.15:
            bullets.append("ROE suggests solid capital efficiency.")
    return f"{name}: " + " ".join(bullets)


# ================= Live News (free RSS) =================
MSFT_IR_RSS = "https://www.microsoft.com/en-us/Investor/rss.xml"
COMPANY_IR_RSS = {
    "MSFT": MSFT_IR_RSS,
    # Add more company IR RSS feeds if you like
}

def fetch_news(ticker: str, long_name: str = "", max_items: int = 6, days=14):
    if not HAVE_FEEDPARSER:
        return [("Install feedparser for live news", "", "", "")]
    q = long_name or ticker
    gnews = f"https://news.google.com/rss/search?q={q}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    feeds = [gnews]
    if ticker in COMPANY_IR_RSS:
        feeds.append(COMPANY_IR_RSS[ticker])

    out = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for url in feeds:
        try:
            fp = feedparser.parse(url)
            for e in fp.entries:
                title = getattr(e, "title", "").strip()
                link = getattr(e, "link", "").strip()
                src = getattr(e, "source", {}).get("title", "") if hasattr(e, "source") else ""
                pub = ""
                try:
                    if hasattr(e, "published_parsed") and e.published_parsed:
                        dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                        if dt >= cutoff:
                            pub = dt.strftime("%Y-%m-%d")
                        else:
                            continue
                except Exception:
                    pass
                if title:
                    out.append((title, src, pub, link))
        except Exception:
            continue

    # Deduplicate & trim
    seen = set(); unique=[]
    for t in out:
        if t[0] not in seen:
            unique.append(t); seen.add(t[0])
    return unique[:max_items]


# ================= Main analysis flow =================
ASSET_CLASSES = ["Stocks","Funds/ETFs","Futures","Forex","Crypto","Indices","Bonds","Economy","Options"]
REGIONS = ["US/Global","India","Singapore"]

DEFAULTS = dict(
    economy="US",
    asset="Stocks",
    region="US/Global",
    ticker="MSFT",
    years=5,
    risk="high",          # low / medium / high
    cash=1000.0,
    portfolio=30000.0,
    desired_pct=5.0,
    hold_months=6
)

def analyze(asset_class: str, region: str, ticker: str, years: int,
            risk: str, cash: float, portfolio: float, desired_pct: float, hold_m: int):
    norm = normalize_symbol(asset_class, region, ticker)
    tk = yf.Ticker(norm)
    info = {}
    try:
        info = tk.get_info() if hasattr(tk, "get_info") else tk.info
    except Exception:
        info = getattr(tk, "info", {}) or {}

    long_name = info.get("longName") or info.get("shortName") or ticker
    currency = info.get("currency", "USD")
    sector = info.get("sector", "—"); industry = info.get("industry", "—")

    hist, providers = history_multi(asset_class, region, norm, years=years)
    if hist is None or hist.empty:
        raise RuntimeError("No historical data from free sources (try a different symbol/region).")

    tscore, tsnap = score_technicals(hist, hold_m)
    fscore = fundamentals_score(info) if asset_class in {"Stocks","Funds/ETFs"} else 0.6
    sent, sbd = market_sentiment(region)
    comp, call = verdict(fscore, tscore, sent, hold_m)
    conviction = round(clamp01(comp) * 100.0, 1)
    rec_pct, cap_pct = allocation_recommendation(conviction, risk, desired_pct)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("="*110)
    print(f"{long_name} ({norm}) | Asset: {asset_class} | Region: {region} | Sector: {sector} | Industry: {industry} | CCY: {currency}")
    print(f"As of: {now} | Data sources: {', '.join(providers) if providers else '—'}")
    print("-"*110)
    print("SCORES")
    print(f"  Fundamentals: {fmt(fscore)} | Technicals: {fmt(tscore)} | Market Sentiment: {fmt(sent)} | Composite: {fmt(comp)}")
    print(f"  Verdict: {call} | Conviction: {conviction:.1f}%")

    print("\nPOSITION SIZING")
    print(f"  Risk level: {risk.upper()} | Max cap per idea: {cap_pct:.2f}% | Desired: {desired_pct:.2f}% → Recommended now: {rec_pct:.2f}%")
    if portfolio and portfolio>0:
        rec_d = round((rec_pct/100.0)*portfolio,2)
        deploy = min(rec_d, cash) if cash is not None else rec_d
        print(f"  Portfolio: ${portfolio:,.2f} | Cash: ${cash:,.2f} | Recommended $: ${rec_d:,.2f} | Deploy now: ${deploy:,.2f}")

    print("\nTECHNICAL SNAPSHOT")
    print(f"  Price: {fmt(tsnap['price'])} {currency} | EMA(short/long): {fmt(tsnap['ema_short'])}/{fmt(tsnap['ema_long'])}")
    print(f"  RSI(14): {fmt(tsnap['rsi'])} | MACD/Signal: {fmt(tsnap['macd'])}/{fmt(tsnap['macd_signal'])} | ADX: {fmt(tsnap['adx'])}")
    print(f"  MFI: {fmt(tsnap['mfi'])} | W%R: {fmt(tsnap['wr'])} | ATR: {fmt(tsnap['atr'])} | BB(mid/up/low): {fmt(tsnap['bb_mid'])}/{fmt(tsnap['bb_up'])}/{fmt(tsnap['bb_lo'])}")

    print("\nWHY THIS CALL (Plain-English)")
    print(" ", explain_choice(long_name, call, dict(rsi=tsnap['rsi']), sent, info))

    print("\nLATEST NEWS (free sources)")
    news_items = fetch_news(ticker=ticker, long_name=long_name, max_items=6, days=14)
    for i, (title, src, pub, link) in enumerate(news_items, 1):
        src_disp = f" — {src}" if src else ""
        pub_disp = f" [{pub}]" if pub else ""
        print(f"  {i:>2}. {title}{src_disp}{pub_disp}")
        if link: print(f"      {link}")

    print("\nNotes")
    print("  • Educational output, not financial advice. Check liquidity, earnings dates, and diversification.")
    print("  • News feed uses Google News RSS + known company IR RSS where available (no API key).")
    print("="*110)


# ================= CLI with Defaults =================
def main():
    print("=== Quant Assistant v6 (All-defaults ready) ===")
    ASSET_CLASSES = ["Stocks","Funds/ETFs","Futures","Forex","Crypto","Indices","Bonds","Economy","Options"]
    REGIONS = ["US/Global","India","Singapore"]
    DEFAULTS = dict(
        economy="US", asset="Stocks", region="US/Global", ticker="MSFT", years=5,
        risk="high", cash=1000.0, portfolio=30000.0, desired_pct=5.0, hold_months=6
    )

    asset = input(f"Asset class {ASSET_CLASSES} [default={DEFAULTS['asset']}]: ").strip() or DEFAULTS["asset"]
    region = input(f"Region {REGIONS} [default={DEFAULTS['region']}]: ").strip() or DEFAULTS["region"]
    ticker = input(f"Ticker [default={DEFAULTS['ticker']}]: ").strip().upper() or DEFAULTS["ticker"]
    years = int(input(f"Years of data [default={DEFAULTS['years']}]: ") or DEFAULTS["years"])
    risk = input("Risk (low/medium/high) [default=high]: ").strip().lower() or DEFAULTS["risk"]
    cash = float(input(f"Available cash [default={DEFAULTS['cash']}]: ") or DEFAULTS["cash"])
    portfolio = float(input(f"Total portfolio value [default={DEFAULTS['portfolio']}]: ") or DEFAULTS["portfolio"])
    desired = float(input(f"Desired allocation % [default={DEFAULTS['desired_pct']}]: ") or DEFAULTS["desired_pct"])
    hold = int(input("Holding period months (1/3/6/12/24) [default=6]: ") or DEFAULTS["hold_months"])

    analyze(asset, region, ticker, years, risk, cash, portfolio, desired, hold)


if __name__ == "__main__":
    main()
