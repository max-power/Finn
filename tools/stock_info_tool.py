from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from tools.tools import handle_tool_error
from tools.stock_price_tool import TickerSymbolSchema

import yfinance as yf

class StockInfoSchema(TickerSymbolSchema):
    """Input for StockInfoTool."""
    key: str = Field(
        description="Which information to look for. Required. 'key' MUST be a SINGLE JSON STRING.",
        default="currentPrice",
        examples=["shortName", "ebitda", "lastDividendDate", "industry", "fullTimeEmployees", "dayLow"],
    )

class StockInfoTool(BaseTool):
    name = "StockInfoTool"
    args_schema = StockInfoSchema
    handle_tool_error = handle_tool_error
    description = """Useful for when you need to find out generell informations about a stock or company. 
        It requires a ticker symbol as first parameter. You MUST obtain a valid ticker symbol first.
        VALID 'key' paramter MUST BE ONE of the following:
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
