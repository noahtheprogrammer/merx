import pandas as pd
import requests

class DataImport:

    def import_yahoo_finance(csv_path: str) -> pd.DataFrame:
        """
        Returns a `DataFrame` object of the imported CSV document.

        :param path: Path to the Yahoo Finance CSV file.
        """

        df = pd.read_csv(csv_path)
        df = df.set_index("Date")
        return df
    
    def import_binance_klines(url: str) -> pd.DataFrame:
        """
        Returns a `DataFrame` object of the inputed API string.

        :param url: URL to access the Binance candlestick JSON data.
        """

        response = requests.get(url=url)
        df = pd.DataFrame(response.json(), columns=['T', 'open', 'high', 'low', 'close', 'volume', 'CT', 'QV', 'N', 'TB', 'TQ', 'I'])
        df["T"] = pd.to_datetime(df["T"], unit="ms")
        df = df.set_index("T")
        return df