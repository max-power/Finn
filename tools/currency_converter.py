from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from forex_python.converter import CurrencyRates
from decimal import Decimal

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables.config import run_in_executor

class CurrencyConverterSchema(BaseModel):
    """Input for Currency Converter Tools."""
    amount: str = Field(title="Amount", description="The amount to convert, ONLY single `amount` value.", examples=['123.45', '10', '1200.95'])
    base:   str = Field(title="Base currency", description="The currency code to convert from", examples=['EUR','USD','JPY'])
    quote:  str = Field(title="Quote currency", description="The currency code to convert to", examples=['EUR','USD','JPY'])


class CurrencyConverterTool(BaseTool):
    name        = "CurrencyConverter"
    description = "Convert from base currency to quote currency. "
    args_schema = CurrencyConverterSchema
    
    def _run(
        self, amount: str, base: str, quote: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return CurrencyRates().convert(base, quote, Decimal(amount))
        
    # async def _arun(
    #     self, amount: str, base: str, quote: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     return await run_in_executor(None, self._run, amount, base, quote)
