from langchain_core.tools import Tool, BaseTool
from tools.stock_price_tool import TickerSymbolSchema
from tools.tools import handle_tool_error

import yfinance as yf

class StockDividendTool(BaseTool):
    name        = "StockDividendTool"
    description = "Useful for when you need to find dividends paid by a company. It returns a pandas dataframe which you can further analyse. It requires a ticker symbol as parameter. You MUST obtain a valid ticker symbol first."
    args_schema = TickerSymbolSchema
    handle_tool_error=handle_tool_error
    
    def _run(self, symbol: str):
        return yf.Ticker(symbol).dividends.to_json(date_format='iso')





# StockDividendTool = Tool.from_function(
#     name        = "StockDividendTool",
#     description = "Useful for when you need to find dividends paid by a company. It returns a pandas dataframe which you can further analyse. It requires a ticker symbol as parameter. You MUST obtain a valid ticker symbol first.",
#     args_schema = TickerSymbolSchema,
#     func        = lambda x: yf.Ticker(x).dividends, #.to_json(),
#     # TODO: returns a pd.DataFrame. maybe parse?
#     handle_tool_error=handle_tool_error,
# )