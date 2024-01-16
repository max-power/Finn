from typing import Optional
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import Tool, BaseTool, StructuredTool, ToolException
from langchain_experimental.tools import PythonREPLTool
from tools.currency_converter import CurrencyConverterTool
from tools.news_classifier import NewsClassifier

from datetime import datetime, timedelta
import yfinance as yf


from json import JSONDecodeError
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


class TickerSymbolSchema(BaseModel):
    """Input for Stock Tools."""
    symbol: str = Field(
        title="Symbol", 
        description="Ticker symbol for stock or index. Required.", 
        examples=['AAPL', 'MSFT', 'BOSS.DE']
    )

date_tool = StructuredTool.from_function(
    name="Datetime",
    func=lambda x: datetime.now().strftime("%A, %B %d, %Y %H:%M:%S"),
    description="Useful for when you are need to find the date and time. Use this tool before any other tool if you are unaware of the current date.",
#    return_direct=True,
    handle_tool_error=handle_tool_error,
)

stock_dividend_tool = Tool.from_function(
    name="Stock-Dividend-Tool",
    func=lambda x: yf.Ticker(x).dividends, #.to_json(),
    description="Useful for when you need to find dividends paid by a company. It returns a pandas dataframe which you can further analyse. It requires a ticker symbol as parameter. You MUST obtain a valid ticker symbol first.",
    handle_tool_error=handle_tool_error,
    args_schema = TickerSymbolSchema,
)

stock_news_tool = Tool.from_function(
    name="Stock-News-Tool",
    func=lambda symbol: yf.Ticker(symbol).news.to_json(date_format='iso'),
    description="Useful for when you need to find news articles about a company or stock. It requires a ticker symbol as parameter. You MUST obtain a valid ticker symbol first.",
    handle_tool_error=handle_tool_error,
    args_schema = TickerSymbolSchema,
)

stock_news_sentiment_tool = Tool.from_function(
    name = "Stock-News-Sentiment-Tool",
    func = lambda x: NewsClassifier().sentiment_for(x),
    description = "Useful for when you need to get a financial sentiment for a news headlines. Expects a news headline or text as parameter.",
    handle_tool_error=handle_tool_error,
)




class StockPriceSchema(TickerSymbolSchema):
    """Input for Stock Price Tools."""
    price_type: Optional[str] = Field(
        title="Price type",
        description="Which price to look for (Open, High, Low, Close, Volume)",
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
        description="Start date in 'YYYY-MM-DD' format (optional).",
        examples=['2024-01-10'],
    )
    end_date: Optional[str] = Field(
        title="End Date",
        description="End date in 'YYYY-MM-DD' format (optional). There MUST be a difference to start_date by at least one day! (start_date + 1)",
        examples=['2024-01-11'],
    )
    
class StockPriceTool(BaseTool):
    name = "Stock-Price-Tool"
    description = "Fetch historical stock data using yFinance. Return: pd.DataFrame: Historical stock data."
    args_schema = StockPriceSchema
    handle_tool_error = handle_tool_error
    #    return_direct = True
    
    def _run(self, symbol: str, price_type: str = 'Close', interval: str = '1d', period: str = '1mo', start_date: str = None, end_date: str = None):
        ticker = yf.Ticker(symbol)

        if start_date and end_date:
            data = ticker.history(interval=interval, start=start_date, end=end_date)
        else:
            data = ticker.history(interval=interval, period=period)

        return data




class StockInfoSchema(TickerSymbolSchema):
    """Input for Stock Information Tool."""
    key:    str = Field(description="Which information to look for. Required. 'key' MUST be a SINGLE JSON STRING.", default="currentPrice")

class StockInfoTool(BaseTool):
    name = "Stock-Information-Tool"
    args_schema = StockInfoSchema
    handle_tool_error = handle_tool_error
    description = """Useful for when you need to find out informations about a stock. 
        It requires a ticker symbol as first parameter. You MUST obtain a valid ticker symbol first.
        The 'key' paramter is ONE of the following keys:
        [address1 (street address), city, state, zip, country, phone, website, industry, 
        industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, 
        fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, 
        shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate,
        maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, 
        regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, 
        exDividendDate, payoutRatio, fiveYearAvgDividendYield, beta, trailingPE, forwardPE, volume, 
        regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask,
        bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months,
        fiftyDayAverage, twoHundredDayAverage, trailingAnnualDividendRate, trailingAnnualDividendYield, 
        currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, 
        sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, 
        heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, 
        impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, 
        mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, 
        lastSplitFactor, lastSplitDate, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, 
        SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, 
        underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, 
        timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, 
        targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, 
        recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, 
        quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, 
        returnOnEquity, grossProfits, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, 
        grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio]
        
        It includes the latest stock price as currentPrice. For older prices use the Stock-Price-Tool.
    """

    def _run(self, symbol: str, key: str = 'currentPrice'):
        return yf.Ticker(symbol).info[key]

#
# class StockNewsSchema(BaseModel):
#     symbol: str = Field(..., description="Ticker symbol for stock/company.")
#
# class StockNewsTool(BaseTool):
#     name = "Stock-News-Tool"
#     args_schema = StockNewsSchema
#     description = "Useful for when you need to find news articles about a company or stock. It requires a ticker symbol as paramter. You MUST obtain a valid ticker symbol first."
#     def _run(self, symbol: str):
#         return yf.Ticker(symbol).news
#


# class StockNewsSentimentSchema(BaseModel):
#     text: str = Field(..., description="News text or headline to analyse")
#
# class StockNewsSentimentTool(BaseTool):
#     args_schema = StockNewsSentimentSchema
#     name = "Financial-News-Sentiment-Tool"
#     description = "Useful for when you need to get a financial sentiment for a news headlines."
#     def _run(self, text: str):
#         return NewsClassifier().sentiment_for(text)


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
# class CurrentStockPriceInput(BaseModel):
#     """Inputs for get_current_stock_price"""
#
#     ticker: str = Field(description="Ticker symbol of the stock")
#
#
# class CurrentStockPriceTool(BaseTool):
#     name = "get_current_stock_price"
#     description = """
#         Useful when you want to get current stock price.
#         You should enter the stock ticker symbol recognized by the yahoo finance
#         """
#     args_schema: Type[BaseModel] = CurrentStockPriceInput
#
#     def _run(self, ticker: str):
#         price_response = get_current_stock_price(ticker)
#         return price_response
#
# current_stock_price_tool = CurrentStockPriceTool()
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


