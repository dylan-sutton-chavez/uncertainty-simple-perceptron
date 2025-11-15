from alpaca.data import StockHistoricalDataClient, StockBarsRequest, StockLatestBarRequest,TimeFrame, TimeFrameUnit

from datetime import datetime, timedelta, UTC

class AlpacaMarkets:
    def __init__(self, alpaca_key: str, alpaca_secret: str, symbol: str):
        """
        Initializes the data client using credentials and sets the specific stock symbol for future use.
        
        Args:
            alpaca_key: str → API key needed for authenticating requests to the Alpaca Markets.
            alpaca_secret: str → Password that confirm and authorize access to the Alpaca API
            symbol: str → Stock ticker symbol (e.g., 'TSLA') for which historical and latest market data.

        Output: 
            None

        Time complexity → o(1)
        """
        self.client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
        self.symbol = symbol

    def historical_market_bars(self, limit_bars: int = 1280, weeks_data_window: int = 4):
        """
        Retrieves 15-minute interval historical price data for the specified symbol over a defined time period.
        
        Args:
            limit_bars: int → Maximum number of 15-minute price bars to retrieve
            weeks_data_window: int = 4 → Defines the window of time in weeks of the bars.

        Output: 
            None

        Time complexity → o(n)
        """
        now_time_utc = datetime.now(UTC)

        from_date = now_time_utc - timedelta(weeks=weeks_data_window + 1)
        to_date = now_time_utc - timedelta(weeks=1)

        timeframe_15_minutes = TimeFrame(amount=15, unit=TimeFrameUnit.Minute)

        request_params = StockBarsRequest(symbol_or_symbols=self.symbol, timeframe=timeframe_15_minutes, start=from_date, end=to_date, limit=limit_bars)

        return self.client.get_stock_bars(request_params)

    def last_minute_bar(self):
        """
        Fetches the most recent, currently available market price bar data for the initialized stock symbol.
        
        Args:
            None

        Output: 
            None

        Time complexity → o(1)
        """
        request_params = StockLatestBarRequest(symbol_or_symbols=self.symbol)

        return self.client.get_stock_latest_bar(request_params)
    
if __name__ == '__main__':
    """
    Securely tests the class functionality by fetching and printing sample market data results.
    
    Time complexity → O(n)

    Initialize → python alpaca_markets.py
    """
    ALPACA_KEY = '1234567890'
    ALPACA_SECRET = '1234567890'
    
    alpaca_markets = AlpacaMarkets(ALPACA_KEY, ALPACA_SECRET, 'TSLA')

    historical_market_bars = alpaca_markets.historical_market_bars()
    print(str(historical_market_bars)[:100])

    last_minute_bar = alpaca_markets.last_minute_bar()
    print(last_minute_bar)