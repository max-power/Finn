from typing import Optional
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from tools.tools import handle_tool_error

import yfinance as yf

from langchain.output_parsers import PandasDataFrameOutputParser

class TickerSymbolSchema(BaseModel):
    """Input for Stock Tools."""
    symbol: str = Field(
        title="Symbol", 
        description="Ticker symbol for stock or index. Required.", 
        examples=['AAPL', 'MSFT', 'BOSS.DE']
    )


# Stock Price Tool #############################################

class StockPriceSchema(TickerSymbolSchema):
    """Input for Stock Price Tools."""
    price_type: Optional[str] = Field(
        title="Price type",
        description="Which price to look for. Allowed values: [Open, High, Low, Close, Volume]",
        default="Close", 
        examples=['Open', 'High', 'Low', 'Close', 'Volume'],
    )
    interval: str = Field(
        title="Interval", 
        description="Data Frequency.", 
        default="1d", 
        examples=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    )
    period: Optional[str] = Field(
        title="Period", 
        description="Historical data period (optional). Don't mix with start_date, end_date!",
        default="1mo",
        examples=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
    )
    start_date: Optional[str] = Field(
        title="Start Date",
        description="Start date in 'YYYY-MM-DD' format (optional). Don't mix with period!",
        examples=['2024-01-10'],
    )
    end_date: Optional[str] = Field(
        title="End Date",
        description="End date in 'YYYY-MM-DD' format (optional). end_date MUST BE GREATER THAN start_date. Don't mix with period!",
        examples=['2024-01-11'],
    )

class StockPriceTool(BaseTool):
    name = "StockPriceTool"
    description = "Fetch stock price data using yFinance. Return: pd.DataFrame: Historical stock data. Input end_date MUST NOT EQUAL start_date."
    args_schema = StockPriceSchema
    handle_tool_error = handle_tool_error
    #    return_direct = True
    
    def _run(self, symbol: str, price_type: str = 'Close', interval: str = '1d', period: str = '1mo', start_date: str = None, end_date: str = None):
        ticker = yf.Ticker(symbol)
        
        from datetime import datetime, timedelta
        def validate_dates(start_str, end_str):
            # Convert string inputs to datetime objects
            start_date = datetime.strptime(start_str, '%Y-%m-%d')
            end_date   = datetime.strptime(end_str,   '%Y-%m-%d')

            # Ensure end_date is at least 1 day after start_date
            if end_date <= start_date:
                end_date = start_date + timedelta(days=1)

            return start_date, end_date

        if start_date and end_date:
            start_date, end_date = validate_dates(start_date, end_date)
            data = ticker.history(interval=interval, start=start_date, end=end_date)
        else:
            data = ticker.history(interval=interval, period=period)
            
        ##############################################
        # return data

        # TODO: pandas DataFrame with or without parser?
        #return PandasDataFrameOutputParser(dataframe=data)
        ####################################################
        
#        return {"price": data[price_type], "currency": ticker.info["currency"]}
        return data[price_type]

