import pandas as pd

class Oscillators:

    def rsi(close: pd.Series, period: int) -> pd.Series:
        """
        Returns a `Series` object containing RSI values for the period and imported DataFrame.

        :param close: `Series` object containing closinng price information.
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

    pass