from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import OpenAI
import pandas as pd

import yfinance as yf
df = yf.Ticker("AAPL").history(period="1y")
pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

pd_agent.invoke("closing price (Close) on the 16. Jan 2024")