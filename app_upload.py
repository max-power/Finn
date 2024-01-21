from chainlit.server import app
from fastapi import Request
from fastapi.responses import HTMLResponse

from langchain.schema.runnable.config import RunnableConfig

from typing import Dict, Any, List
from chainlit.element import Element
import chainlit as cl

from finn import Finn
from langchain_openai import ChatOpenAI, OpenAI

from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate

from langchain.tools import tool
from typing import List, Optional
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ChatMessageHistory

# LLM Cache
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
#set_llm_cache(SQLiteCache(database_path="db/app_upload_llm_cache.db"))

from datetime import datetime
from utils.file_loader import FileLoader

combo_template = f"""\n\nHuman:
You (named Finn) are the most seasoned financial analyst and investment advisor with expertise
in stock market analysis and investment strategies. You are skilled in sifting through
news, company announcements, market sentiments, income statement, balance sheet, and more.
You combine various analytical insights to formulate strategic investment advice.

To ensure the accuracy of information, You will lookup the current date using the `DateTool`
to utilize it to retrieve the latest stock price data. You have access to finance data and
company information using the appropiate tools. You also can conduct web searches and refer 
to Wikipedia for comprehensive information. You will ensure to include the sentiment analysis 
and when providing news headlines. 

You would be presented with a problem. First understand the problem and devise a plan to solve the problem. 
Please output the plan starting with the header 'Plan:' and then followed by a numbered list of steps.        
Ensure the plan has the minimum amount of steps needed to solve the problem. Do not include unnecessary steps.

<instructions>
These are guidance on when to use a tool to solve a task, follow them strictly:
1. To lookup the current date and time use the "DateTool". It is necessary for financial analysis to know the current date.
2. For the tool that specifically focuses on stock price data, use "StockPriceTool".    
3. For financial information lookup that covers various financial data like company's finance, performance or any other information pertaining a company beyond stocks, use the "StockInfoTool".
4. For news related to stocks or companies use the "StockNewsTool" and the "StockNewsSentimentTool" to get a sentiment.
5. When you need to analyze sentiment of a topic, use "StockNewsSentimentTool". 
6. For deeper financial insides into a company or stock, use this Tools: "StockRecommendationTool", "StockCashFlowTool", "StockIncomeStatementTool", "StockBalanceSheetTool", "StockDividendTool.
7. When you need to perform mathematical calculation use the "Calculator". To perform currency convertion use the "CurrencyConverterTool".
8. The human might have uploaded importent document in the VectorStore. Use the "VectorStoreDocumentSearch" to search uploaded documents.
9. You can conduct web searches using DuckDuckGo, as well as refer to Wikipedia for generell Informations.
10. If necessary use the "human" tool to ask the human for instructions.
</instructions>

If a tool is not working change the input or try a different tool. Retry only ONCE.

When asked for investment advise you will providing in-depth explanations on company
information, stock prices, news, balance sheet, income statements, cash flow, recommendations 
and other sources to create a comprehensive due diligence report. 
Present the investment report to the human user in a comprehensive and well-structured format. 

\n\nAssistant:"""

DATE_TEMPLATE = f"Today is {datetime.now().strftime('%A, %B %d, %Y %H:%M:%S')}."

from prompts.finn_prompt import SYSTEM_PROMPT, HUMAN_PROMPT
from prompts.stock_analyse_prompt import STOCK_ANALYSE_PROMPT2

@cl.on_chat_start
async def on_chat_start():
    finn     = Finn()
    #model    = finn.chat_llm
    model    = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, cache=True)
    planner  = load_chat_planner(model)

    system_message_prompt    = SystemMessagePromptTemplate.from_template(combo_template)
    human_message_prompt     = planner.llm_chain.prompt.messages[1]
    
    planner.llm_chain.prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    history  = ChatMessageHistory()
    memory   = ConversationBufferMemory(memory_key="chat_history", chat_memory=history, return_messages=True)
    
    executor = load_agent_executor(model, finn.tools, verbose=True)
    agent    = PlanAndExecute(planner=planner, executor=executor, verbose=True, max_iterations=2, memory=memory)

    cl.user_session.set("finn", finn)
    cl.user_session.set("runnable", agent)

    

@cl.on_message
async def on_message(message: cl.Message):
    runnable        = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer  = True,
        answer_prefix_tokens = ["Final", "Answer"]
    )])
    
    msg = cl.Message(content="", disable_feedback=True)
    await msg.send()

    if message.elements:
        msg.content = f"Processing files …"
        await msg.update()
        await process_files(message.elements)
        # Let the user know that the system is ready
        msg.content = f"Processing done. You can now ask questions!"
        await msg.update()

    # send response
    response = await runnable.ainvoke({"input": message.content}, config=runnable_config)
    await msg.stream_token(response["output"])
    # async with cl.Step(type="run", name="QA Assistant"):
    #     async for chunk in runnable.astream({"input": message.content}, config=runnable_config):
    #         await msg.stream_token(chunk)



@cl.step(name="File Upload", type="run", root=False)
async def process_files(files: List[Element]):
    finn      = cl.user_session.get("finn")
    documents = [FileLoader(file).content for file in files]
    finn.vectorstore.add_documents(documents, ids=None) #TODO: what does ids do?
    files_str = "1 file" if len(files)==1 else f"{len(files)} files"
    return f"{files_str} added to the Vector store."
