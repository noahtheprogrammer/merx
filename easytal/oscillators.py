import pandas as pd
from easytal.overlays import ema

def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Relative Strength Index.
    """

    delta = close.diff()
    up = delta.clip(lower=0)
    down = delta.clip(upper=0).abs()

    upper_ema = up.ewm(com = period - 1, adjust=False, min_periods=period).mean()
    lower_ema = down.ewm(com = period - 1, adjust=False, min_periods=period).mean()
    rsi = upper_ema / lower_ema
    rsi = 100 - (100/(1 + rsi))
    
    return rsi

def macd(close: pd.Series, slow: int, fast: int, period: int) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the histogram, signal line, and MACD.
    """

    slow_ema = ema(close, slow)
    fast_ema = ema(close, fast)

    macd = fast_ema - slow_ema
    signal = ema(macd, period)
    histogram = macd - signal

    columns=["MACD", "Signal", "Histogram"]
    macd_df = pd.DataFrame(columns=columns)

    macd_df["MACD"] = macd
    macd_df["Signal"] = signal
    macd_df["Histogram"] = histogram

    return macd_df

def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Returns a `Series` object containing the True Range.
    """
    
    arr = []

    for x in range(1, len(close)):
        
        val = max(high[x] - low[x], abs(high[x] - close[x-1]), abs(low[x] - close[x-1]))
        arr.append(val)
    
    tr_series = pd.Series(arr)

    return tr_series

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Average True Range.
    """
    
    arr = tr(high, low, close)
    atr_series = arr.rolling(period).mean()

    return atr_series