from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from tools.tools import handle_tool_error
from tools.stock_price_tool import TickerSymbolSchema

import yfinance as yf

class StockNewsTool(BaseTool):
    name        = "StockNewsTool"
    description = "Useful for when you are need to find the date and or time. Remember allowed JSON input keys 'input' and 'format' (both optional)."
    args_schema = TickerSymbolSchema
    handle_tool_error=handle_tool_error
    
    def _run(self, symbol: str):
        return yf.Ticker(symbol).news # .to_json(date_format='iso'),
