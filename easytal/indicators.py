import pandas as pd
from math import sqrt

def rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing RSI values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the RSI.
    """

    delta = close.diff()
    up = delta.clip(lower=0)
    down = delta.clip(upper=0).abs()

    upper_ema = up.ewm(com = period - 1, adjust=False, min_periods=period).mean()
    lower_ema = down.ewm(com = period - 1, adjust=False, min_periods=period).mean()
    rsi = upper_ema / lower_ema
    rsi = 100 - (100/(1 + rsi))
    
    return rsi

def sma(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing SMA values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the SMA.
    """

    sma = close.rolling(period).mean()

    return sma

def ema(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing EMA values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the EMA.
    """

    ema = close.ewm(span=period, adjust=False).mean()

    return ema

def dema(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing DEMA values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the DEMA.
    """
    
    primary_ema = ema(close, period)
    secondary_ema = ema(primary_ema, period)
    dema = (primary_ema * 2) - secondary_ema

    return dema

def tema(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing TEMA values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the TEMA.
    """
    
    primary_ema = ema(close, period)
    secondary_ema = ema(primary_ema, period)
    tertiary_ema = ema(secondary_ema, period)
    tema = (primary_ema * 3) - (secondary_ema * 3) + tertiary_ema

    return tema

def wma(close: pd.Series, period: int) -> pd.Series:
    """
    Returns a `Series` object containing WMA values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the WMA.
    """
    
    wma = close.rolling(period).apply(lambda x: x[::-1].cumsum().sum() * 2 / period / (period + 1))

    return wma

def hma(close: pd.Series, period: int)  -> pd.Series:
    """
    Returns a `Series` object containing HMA values for the period and imported series.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the HMA.
    """

    primary_wma = wma(close, round(period/2))
    secondary_wma = wma(close, period)

    raw_hma = (2 * primary_wma) - secondary_wma
    hma = wma(raw_hma, round(sqrt(period)))

    return hma

def bollinger_bands(close: pd.Series, period: int) -> pd.Series:
    """
    Returns both the upper and lower Bollinger Band values as `Series` objects.

    :param close: `Series` object containing closing price information.
    :param period: Integer value over which to calculate the Bollinger Bands.
    """

    simple = sma(close, period)
    std = close.rolling(period).std()

    upper_bband = simple + std * 2
    lower_bband = simple - std * 2

    return upper_bband, lower_bband

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Returns the pivot point, first support, second support, first resistance, and second resistance in a `DataFrame` object.

    :param high: `Series` object containing high price information.
    :param low: `Series` object containing low price information.
    :param close: `Series` object containing closing price information.
    """

    columns=["Pivot", "First Support", "Second Support", "First Resistance", "Second Resistance"]
    pivot_df = pd.DataFrame(columns=columns)

    for x in range(0, len(close)):

        point = (high[x] + low[x] + close[x])/3

        first_sprt = (point * 2) - high[x]
        second_sprt = point - (high[x] - low[x])
        first_res = (point * 2) - low[x]
        second_res = point + (high[x] - low[x])

        pivot_df.loc[len(pivot_df)] = [point, first_sprt, second_sprt, first_res, second_res]

    return pivot_df