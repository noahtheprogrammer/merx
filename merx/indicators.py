import pandas as pd
from math import sqrt

def sma(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Simple Moving Average.
    """

    sma = close.rolling(period).mean()

    return sma

def ema(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Exponential Moving Average.
    """

    ema = close.ewm(span=period, adjust=False).mean()

    return ema

def dema(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Double Exponential Moving Average.
    """
    
    primary_ema = ema(close, period)
    secondary_ema = ema(primary_ema, period)
    dema = (primary_ema * 2) - secondary_ema

    return dema

def tema(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Triple Exponential Moving Average.
    """
    
    primary_ema = ema(close, period)
    secondary_ema = ema(primary_ema, period)
    tertiary_ema = ema(secondary_ema, period)
    tema = (primary_ema * 3) - (secondary_ema * 3) + tertiary_ema

    return tema

def wma(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Weighted Moving Average.
    """
    
    wma = close.rolling(period).apply(lambda x: x[::-1].cumsum().sum() * 2 / period / (period + 1))

    return wma

def hma(close: pd.Series, period: int)  -> pd.Series:
    """
    Returns a `Series` object containing the Hull Moving Average.
    """

    primary_wma = wma(close, round(period/2))
    secondary_wma = wma(close, period)

    raw_hma = (2 * primary_wma) - secondary_wma
    hma = wma(raw_hma, round(sqrt(period)))

    return hma

def ma_envelope(moving_average: pd.Series, multiplier: float):
    """
    Returns a `DataFrame` object containing the upper and lower moving average bands.
    """

    upper_ma = moving_average + moving_average * multiplier
    lower_ma = moving_average - moving_average * multiplier

    envelope = pd.concat([upper_ma, lower_ma], axis=1)
    envelope.columns = ["upper", "lower"]
    envelope.index.name = None

    return envelope

def standard_pivot(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the standard pivot point, two support levels, and two resistance levels.
    """

    date_arr = close.index.tolist()
    points = pd.DataFrame(index=date_arr)

    points["pivot"] = (high + low + close)/3
    points = points.assign(sup1=lambda x: (x["pivot"] * 2 - high[x.index]))
    points = points.assign(sup2=lambda x: (x["pivot"] - (high[x.index] - low[x.index])))
    points = points.assign(res1=lambda x: (x["pivot"] * 2 - low[x.index]))
    points = points.assign(res2=lambda x: (x["pivot"] + (high[x.index] - low[x.index])))

    return points

def fibonacci_pivot(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the Fibonacci pivot point, three support levels, and three resistance levels.
    """

    date_arr = close.index.tolist()
    fibb = pd.DataFrame(index=date_arr)

    fibb["pivot"] = (high + low + close)/3
    fibb = fibb.assign(sup1=lambda x: x["pivot"] - 0.382 * (high[x.index] - low[x.index]))
    fibb = fibb.assign(sup2=lambda x: x["pivot"] - 0.618 * (high[x.index] - low[x.index]))
    fibb = fibb.assign(sup3=lambda x: x["pivot"] - 1 * (high[x.index] - low[x.index]))
    fibb = fibb.assign(res1=lambda x: x["pivot"] + 0.382 * (high[x.index] - low[x.index]))
    fibb = fibb.assign(res2=lambda x: x["pivot"] + 0.618 * (high[x.index] - low[x.index]))
    fibb = fibb.assign(res3=lambda x: x["pivot"] + 1 * (high[x.index] - low[x.index]))

    return fibb

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

def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Returns a `Series` object containing the True Range.
    """

    tr = pd.Series()
    tr = pd.concat([high, close.shift()], axis=1).max(axis=1) - pd.concat([low, close.shift()], axis=1).min(axis=1)

    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Average True Range.
    """

    date_arr = close.index.tolist()
    date_arr.pop(0)
    
    arr = tr(high, low, close)
    atr = arr.rolling(period).mean()
    atr = atr.set_axis(date_arr)

    return atr

def er(close: pd.Series) -> pd.Series:
    """
    Returns a `Series` object containing the Efficiency Ratio.
    """

    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(10).sum()
    er = change/volatility

    return er

def chandelier(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing long and short Chandelier exits.
    """

    long_exit = high.rolling(22).max() - (atr(high, low, close, 22) * 3)
    short_exit = low.rolling(22).max() + (atr(high, low, close, 22) * 3)

    chandelier = pd.DataFrame(columns=["long", "short"])
    chandelier["long"], chandelier["short"] = long_exit, short_exit

    return chandelier

def macd(close: pd.Series, slow: int, fast: int, period: int) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the histogram, signal line, and MACD.
    """

    slow_ema = ema(close, slow)
    fast_ema = ema(close, fast)

    macd = fast_ema - slow_ema
    signal = ema(macd, period)
    histogram = macd - signal

    macd = pd.DataFrame(columns=["macd", "signal", "histogram"])
    macd["macd"], macd["signal"], macd["histogram"]  = macd, signal, histogram
    macd.index.name = None

    return macd

def bollinger_bands(close: pd.Series, period: int) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the upper and lower Bollinger Bands.
    """

    simple = sma(close, period)
    std = close.rolling(period).std()

    upper = simple + std * 2
    lower = simple - std * 2

    bband = pd.DataFrame(columns=["upper", "lower"])
    bband["upper"], bband["lower"] = upper, lower
    bband.index.name = None
    
    return bband