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

    envelope_df = pd.concat([upper_ma, lower_ma], axis=1)
    envelope_df.columns = ["upper", "lower"]
    envelope_df.index.name = None

    return envelope_df

def bollinger_bands(close: pd.Series, period: int) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the upper and lower Bollinger Bands.
    """

    simple = sma(close, period)
    std = close.rolling(period).std()

    upper_bband = simple + std * 2
    lower_bband = simple - std * 2

    bband_df = pd.concat([upper_bband, lower_bband], axis=1)
    bband_df.columns = ["upper", "lower"]
    bband_df.index.name = None

    return bband_df

def standard_pivot(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the standard pivot point, two support levels, and two resistance levels.
    """

    columns = ["pivot", "support_1", "support_2", "resistance_1", "resistance_2"]
    pivot_df = pd.DataFrame(columns=columns)

    for x in range(0, len(close)):

        point = (high[x] + low[x] + close[x])/3

        first_sprt = (point * 2) - high[x]
        second_sprt = point - (high[x] - low[x])
        first_res = (point * 2) - low[x]
        second_res = point + (high[x] - low[x])

        pivot_df.loc[len(pivot_df)] = [point, first_sprt, second_sprt, first_res, second_res]
    
    date_arr = close.index.tolist()
    pivot_df = pivot_df.set_axis(date_arr)

    return pivot_df

def fibonacci_pivot(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the Fibonacci pivot point, three support levels, and three resistance levels.
    """

    columns = ["pivot", "support_1", "support_2", "support_3", "resistance_1", "resistance_2", "resistance_3"]
    fib_df = pd.DataFrame(columns=columns)

    for x in range(0, len(close)):

        point = (high[x] + low[x] + close[x])/3

        first_sprt = point - 0.382 * (high[x] - low[x])
        second_sprt = point - 0.618 * (high[x] - low[x])
        third_sprt = point - 1 * (high[x] - low[x])
        first_res = point + 0.382 * (high[x] - low[x])
        second_res = point + 0.618 * (high[x] - low[x])
        third_res = point + 1 * (high[x] - low[x])

        fib_df.loc[len(fib_df)] = [point, first_sprt, second_sprt, third_sprt, first_res, second_res, third_res]

    date_arr = close.index.tolist()
    fib_df = fib_df.set_axis(date_arr)

    return fib_df

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
    
    arr = []
    date_arr = close.index.tolist()
    date_arr.pop(0)

    for x in range(1, len(close)):
        
        val = max(high[x] - low[x], abs(high[x] - close[x-1]), abs(low[x] - close[x-1]))
        arr.append(val)
    
    tr_series = pd.Series(arr)
    tr_series = tr_series.set_axis(date_arr)

    return tr_series

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing the Average True Range.
    """

    date_arr = close.index.tolist()
    date_arr.pop(0)
    
    arr = tr(high, low, close)
    atr_series = arr.rolling(period).mean()
    atr_series = atr_series.set_axis(date_arr)

    return atr_series

def chandelier(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing long and short Chandelier exits.
    """

    long_exit = high.rolling(22).max() - (atr(high, low, close, 22) * 3)
    short_exit = low.rolling(22).max() + (atr(high, low, close, 22) * 3)

    chandelier_df = pd.DataFrame(columns=["long", "short"])
    chandelier_df["long"], chandelier_df["short"] = long_exit, short_exit

    return chandelier_df

def macd(close: pd.Series, slow: int, fast: int, period: int) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the histogram, signal line, and MACD.
    """

    slow_ema = ema(close, slow)
    fast_ema = ema(close, fast)

    macd = fast_ema - slow_ema
    signal = ema(macd, period)
    histogram = macd - signal

    macd_df = pd.DataFrame(columns=["macd", "signal", "histogram"])
    macd_df["macd"], macd_df["signal"], macd_df["histogram"]  = macd, signal, histogram

    return macd_df