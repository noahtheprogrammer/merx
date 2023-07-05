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

def bollinger_bands(close: pd.Series, period: int) -> pd.Series:
    """
    Returns two `Series` objects containing the upper and lower Bollinger Bands.
    """

    simple = sma(close, period)
    std = close.rolling(period).std()

    upper_bband = simple + std * 2
    lower_bband = simple - std * 2

    bband_df = pd.concat([upper_bband, lower_bband], axis=1)
    bband_df.columns = ["upper", "lower"]
    bband_df.index.name = None

    return bband_df

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns a `DataFrame` object containing the pivot point, first support/resistance, and second support/resistance.
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