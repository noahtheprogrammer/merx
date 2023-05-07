import pandas as pd

class Oscillators:

    def __init__(self) -> None:
        pass

    def rsi(self, close: pd.Series, period: int) -> pd.Series:
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
    
class Overlays:

    def __init__(self) -> None:
        pass

    def sma(self, close: pd.Series, period: int) -> pd.Series:
        """
        Returns a `Series` object containing SMA values for the period and imported series.

        :param close: `Series` object containing closing price information.
        :param period: Integer value over which to calculate the SMA.
        """

        sma = close.rolling(period).mean()

        return sma

    def ema(self, close: pd.Series, period: int) -> pd.Series:
        """
        Returns a `Series` object containing EMA values for the period and imported series.

        :param close: `Series` object containing closing price information.
        :param period: Integer value over which to calculate the EMA.
        """

        ema = close.ewm(span=period, adjust=False).mean()

        return ema

    def bollinger_bands(self, close: pd.Series, period: int) -> pd.Series:
        """
        Returns both the upper and lower Bollinger Band values as `Series` objects.

        :param close: `Series` object containing closing price information.
        :param period: Integer value over which to calculate the Bollinger Bands.
        """

        sma = self.sma(close, period)
        std = close.rolling(period).std()

        upper_bband = sma + std * 2
        lower_bband = sma - std * 2

        return upper_bband, lower_bband