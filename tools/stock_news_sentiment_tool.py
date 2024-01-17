from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from tools.tools import handle_tool_error

from utils.news_classifier import NewsClassifier

class StockNewsSentimentSchema(BaseModel):
    text: str = Field(description="News text or headline to analyse")

class StockNewsSentimentTool(BaseTool):
    name        = "StockNewsSentimentTool"
    description = "Useful for when you need to get a financial sentiment for a news headlines. Input is a news headline or text."
    args_schema = StockNewsSentimentSchema
    handle_tool_error=handle_tool_error

    def _run(self, text: str):
        return NewsClassifier(model="yiyanghkust/finbert-tone").sentiment_for(text)


# from langchain_core.tools import Tool
#
# StockNewsSentimentTool = Tool.from_function(
#     func        = lambda str: NewsClassifier().sentiment_for(str),
#     name        = "StockNewsSentimentTool",
#     description = "Useful for when you need to get a financial sentiment for a news headlines. Input is a news headline or text.",
#     handle_tool_error=handle_tool_error,
# )