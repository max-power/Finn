from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool

class CurrencyConverterSchema(BaseModel):
    """Input for Currency Converter Tools."""
    amount: str = Field(title="Value", description="The amount to convert", examples=['123.45', '10', '1200.95'])
    base:   str = Field(title="Base currency", description="The currency code to convert from", examples=['EUR','USD','JPY'])
    quote:  str = Field(title="Quote currency", description="The currency code to convert to", examples=['EUR','USD','JPY'])

from forex_python.converter import CurrencyRates
from decimal import Decimal

class CurrencyConverterTool(BaseTool):
    name = "Currency Converter"
    description = "Convert from base currency to quote currency."
    args_schema = CurrencyConverterSchema
    
    def _run(self, amount: str, base: str, quote: str):
        return CurrencyRates().convert(base, quote, Decimal(amount))
