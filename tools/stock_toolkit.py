from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_core.tools import Tool, StructuredTool
# async
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables.config import run_in_executor
#from langchain.output_parsers import PandasDataFrameOutputParser

# error handling
from tools.tools import handle_tool_error

# finance
import yfinance as yf
import json
from tabulate import tabulate

# Base Schema for all StockTool
class StockSymbolSchema(BaseModel):
    """Input for Stock Tools."""
    symbol: str = Field(
        title="Symbol", 
        description="Ticker symbol for stock or index. Required.", 
        examples=['AAPL', 'MSFT', 'BOSS.DE']
    )

# Base StockTool for inheritance.
class StockBaseTool(BaseTool):
    """Base for StockTools."""
    args_schema = StockSymbolSchema
    handle_tool_error = handle_tool_error

    def _run(
        self, symbol: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> json:
        """Use the tool"""
        raise NotImplementedError(f"Function not implemented in {self.name}")

    
    async def _arun(
        self, symbol: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> json:
        """Use the tool asynchronously."""
        return await run_in_executor(None, self._run, symbol)


# TOOLS ###################################################
 
StockRecommendationTool = StructuredTool(
    func        = lambda symbol: yf.Ticker(symbol).recommendations.to_json(date_format='iso'),
    name        = "StockRecommendationTool",
    description = "Useful for when you need purchase recommendations for a stock.",
    args_schema = StockSymbolSchema,
    handle_tool_error=handle_tool_error,
    ##coroutine=
)

StockCashFlowTool = StructuredTool(
    func        = lambda symbol: yf.Ticker(symbol).cash_flow.to_json(date_format='iso'),
    name        = "StockCashFlowTool",
    description = "Useful for financial analysis when you need to get a cash flow report for a company",
    args_schema = StockSymbolSchema,
    handle_tool_error=handle_tool_error,
)

StockIncomeStatementTool = StructuredTool(
    func        = lambda symbol: yf.Ticker(symbol).income_stmt.to_json(date_format='iso'),
    name        = "StockIncomeStatementTool",
    description = "Useful for financial analysis when you need to get the income statement report for a company",
    args_schema = StockSymbolSchema,
    handle_tool_error=handle_tool_error,
)

StockBalanceSheetTool = StructuredTool(
    func        = lambda symbol: yf.Ticker(symbol).balance_sheet.to_json(date_format='iso'),
    name        = "StockBalanceSheetTool",
    description = "Useful for financial analysis when you need to get the balance sheet for a company",
    args_schema = StockSymbolSchema,
    handle_tool_error=handle_tool_error,
)


# Stock Dividend Tool #############################################
class StockDividendTool(StockBaseTool):
    name        = "StockDividendTool"
    description = "Useful for when you need to find dividends paid by a company. It returns a pandas dataframe which you can further analyse. It requires a ticker symbol as parameter. You MUST obtain a valid ticker symbol first."

    def _run(self, symbol: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        return yf.Ticker(symbol).dividends.to_json(date_format="iso")


# Stock News Tool #############################################
class StockNewsTool(StockBaseTool):
    name        = "StockNewsTool"
    description = "Useful for when you are need news about a company."
    args_schema = StockSymbolSchema
    handle_tool_error = handle_tool_error

    def _run(self, symbol: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> json:
        return json.dumps(yf.Ticker(symbol).news)


# Stock News Classifier Tool #############################################
from utils.news_classifier import NewsClassifier

class StockNewsSentimentSchema(BaseModel):
    text: str = Field(description="News text or headline to analyse")

class StockNewsSentimentTool(BaseTool):
    name        = "StockNewsSentimentTool"
    description = "Useful for when you need to get a financial sentiment for a news headlines. Input is a news headline or text."
    args_schema = StockNewsSentimentSchema
    handle_tool_error=handle_tool_error

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        s = NewsClassifier(model="yiyanghkust/finbert-tone").sentiment_for(text)
        return json.dumps(s)
        
    async def _arun(self, text: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        return await run_in_executor(None, self._run, text, run_manager)

# StockNewsSentimentTool = Tool.from_function(
#     func        = lambda str: NewsClassifier().sentiment_for(str),
#     name        = "StockNewsSentimentTool",
#     description = "Useful for when you need to get a financial sentiment for a news headlines. Input is a news headline or text.",
#     handle_tool_error=handle_tool_error,
# )


def ticker_exists(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Attempt to fetch information to confirm the symbol exists
        info = ticker.info
        return True
    except ValueError:
        # If the symbol is not found, a ValueError will be raised
        return False

# Stock Info Tool #############################################
class StockInfoSchema(StockSymbolSchema):
    """Input for StockInfoTool."""
    key: Optional[str] = Field(
        description = "Which information to look for. Required! 'ALL' return all information at once. Examples: 'longBusinessSummary' or 'payoutRatio'. If `key` is present it must be a SINGLE JSON STRING.",
        examples    = ["currentPrice", "shortName", "ebitda", "lastDividendDate", "industry", "fullTimeEmployees", "dayLow"],
    )
StockInfoSchema = StockSymbolSchema

class StockInfoTool(StockBaseTool):
    args_schema = StockInfoSchema
    name        = "StockInfoTool"
    description = """Useful for when you need to find out general informations about a stock or company. """
    """It REQUIRES a ticker 'symbol' AND 'key' which MUST be one of the following:
        [address1, city, state, zip, country, phone, website, industry, 
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
    """
    handle_tool_error = handle_tool_error
    
    def _run(self, symbol: str, key: str = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> json:
        #return tabulate(yf.Ticker(symbol).info.items())   
        try: 
            x = yf.Ticker(symbol)
            return json.dumps(x.info)
            if 'ALL' == str(key).upper():
                #return tabulate(x.info.items())
                return json.dumps(x.info)
            else:
                return json.dumps({'text': x.info[key]})
            
        except KeyError:
            return f"!KEY ERROR: {repr(key)} does not exist!"
        except ValueError:
            # If the symbol is not found, a ValueError will be raised
            return f"!ERROR: no information available for {symbol}!"
            #return f"!ERROR: {symbol} does not exist!"

    async def _arun(self, symbol: str, key: str = None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> json:
        return await run_in_executor(None, self._run, symbol, key, run_manager)


# Stock Price Tool #############################################


class StockPriceSchema(StockSymbolSchema):
    """Input for Stock Price Tools."""
    price_type: Optional[List[str]] = Field(
        title="Price type",
        description="Which prices to return. Valid values: 'Open', 'High', 'Low', 'Close', 'Volume'. Always use array notation.",
        default=['Close'],
        examples=[['Open'], ['High', 'Low', 'Close', 'Volume']],
    )
    interval: str = Field(
        title="Interval", 
        description=" Valid input: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo. Intraday data cannot extend last 60 days", 
        default="1d", 
        examples=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    )
    period: Optional[str] = Field(
        title="Period",
        description="Historical data period (optional). Either Use period parameter or use start and end. Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max .",
        default="1mo",
        examples=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
    )
    start_date: Optional[str] = Field(
        title="Start Date",
        description="Start date string (YYYY-MM-DD) (optional). IMPORTANT: NEVER mix with arg `period`!",
        examples=['2024-01-10'],
    )
    end_date: Optional[str] = Field(
        title="End Date",
        description="End date string (YYYY-MM-DD) (optional). IMPORTANT: `end_date` MUST BE GREATER THAN `start_date`.",
        examples=['2024-01-11'],
    )

class StockPriceTool(StockBaseTool):
    name        = "StockPriceTool"
    description = "Fetch stock price data using yFinance. Returns: Historical stock data."
    args_schema = StockPriceSchema
    handle_tool_error = handle_tool_error

    def _run(
        self, 
        symbol: str, 
        price_types: List[str] = ['Close'], 
        interval: str = '1d', 
        period: str = '1mo', 
        start_date: str = None, 
        end_date: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> json:

        ticker = yf.Ticker(symbol)

        if start_date and end_date:
            if end_date == start_date:
                return {'error': 'ERROR: Invalid input. `start_date` MUST BE BEFORE `end_date`'}
            data = ticker.history(interval=interval, start=start_date, end=end_date)
        else:
            data = ticker.history(interval=interval, period=period)
            
        ##############################################
        # TODO: pandas DataFrame with or without parser?
        #return PandasDataFrameOutputParser(dataframe=data) or pandasLLM
        ####################################################
        return data[price_types].to_json(date_format='iso', double_precision=2)


    async def _arun(self, 
        symbol: str, 
        price_types: List[str] = ['Close'], 
        interval: str = '1d', 
        period: str = '1mo',
        start_date: str = None,
        end_date: str = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> json:
        """Use the tool asynchronously."""
        return await run_in_executor(None, self._run, symbol, price_types, interval, period, start_date, end_date, run_manager)


# TOOLKIT ####################################################
class StockToolkit(BaseToolkit):
    """Toolkit for Stocks. Mostly yFinance tools."""

    tools = [
        StockPriceTool(),
        StockInfoTool(),
        StockDividendTool(),
        StockNewsTool(),
        StockNewsSentimentTool(),

        StockRecommendationTool,
        StockCashFlowTool,
        StockIncomeStatementTool,
        StockBalanceSheetTool,
    ]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
