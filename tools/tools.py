# Error Handling #####################################################################
from json import JSONDecodeError
from langchain_core.tools import ToolException

def handle_tool_error(error: ToolException) -> str:
    """ Handles tool exceptions. """

    if error==JSONDecodeError:
        return "Reformat in JSON and try again"
    elif error.args[0].startswith("Too many arguments to single-input tool"):
        return "Format in a SINGLE JSON STRING. DO NOT USE MULTI-ARGUMENTS INPUT."
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool.")


# Junkyard ##########################################################################

# StockPlotTool = Tool.from_function(
#
# )
#
# def plot_stock_price(data):
# #  data = yf.Ticker(ticker).history(period='1y')
#   plt.figure(figsize=(10,5))
#   plt.plot(data.index,data.Close)
#   plt.title(f'{ticker} Stock Price Over Last Year')
#   plt.xlabel('Date')
#   plt.ylabel('Stock Price ($)')
#   plt.grid(True)
#   plt.savefig('stock.png')
#   plt.close()



########################################################################
# https://github.com/truera/trulens/blob/9db4b0c70cddd3c7ed1ff4566e928ff329e4d135/trulens_eval/examples/expositional/frameworks/langchain/langchain_agents.ipynb#L235
# def get_current_stock_price(ticker):
#     """Method to get current stock price"""
#
#     ticker_data = yf.Ticker(ticker)
#     recent = ticker_data.history(period="1d")
#     return {"price": recent.iloc[0]["Close"], "currency": ticker_data.info["currency"]}
#
#
# def get_stock_performance(ticker, days):
#     """Method to get stock price change in percentage"""
#
#     past_date = datetime.today() - timedelta(days=days)
#     ticker_data = yf.Ticker(ticker)
#     history = ticker_data.history(start=past_date)
#     old_price = history.iloc[0]["Close"]
#     current_price = history.iloc[-1]["Close"]
#     return {"percent_change": ((current_price - old_price) / old_price) * 100}
# Make custom tools
#
# class StockPercentChangeInput(BaseModel):
#     """Inputs for get_stock_performance"""
#
#     ticker: str = Field(description="Ticker symbol of the stock")
#     days: int = Field(description="Timedelta days to get past date from current date")
#
#
# class StockPerformanceTool(BaseTool):
#     name = "get_stock_performance"
#     description = """
#         Useful when you want to check performance of the stock.
#         You should enter the stock ticker symbol recognized by the yahoo finance.
#         You should enter days as number of days from today from which performance needs to be check.
#         output will be the change in the stock price represented as a percentage.
#         """
#     args_schema: Type[BaseModel] = StockPercentChangeInput
#
#     def _run(self, ticker: str, days: int):
#         response = get_stock_performance(ticker, days)
#         return response
#
# stock_performance_tool = StockPerformanceTool()


