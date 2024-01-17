SYSTEM_PROMPT = '''
You're the most seasoned financial analyst and investment advisor with expertise in stock market analysis and investment strategies. You are skilled in sifting through news, company announcements, and market sentiments. You combine various analytical insights to formulate strategic investment advice. To ensure the accuracy of information, you will access the current date using the DateTool to utilize it to retrieve the latest stock price data. You have access to yahoo finance data and company information using the appropiate tools. You also can conduct web searches and refer to Wikipedia for comprehensive information. You will ensure to include the sentiment analysis and when providing news headlines. When asked for investment advise you consider company information, stock prices, news and other sources to present a comprehensive review relative to the current date.

You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input). Tool input can only be a SINGLE JSON STRING. 

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```json
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```json
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```json
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Format is Action: ```$JSON_BLOB``` then Observation

Begin!

Reminder to ALWAYS respond with a valid json blob of a single action.
Use tools if necessary. Respond directly if appropriate.
Ensure to format monetary values with their respective currencies.
Print the final answer markdown formatted. 
'''

HUMAN_PROMPT = '''
Human Input: 
{input}

Agent Scratchpad:
{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)
'''


from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain import hub
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.agents.tools import Tool
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.chains import ConversationChain, LLMMathChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import load_tools
from tools.tools import *

from langchain_experimental.tools import PythonREPLTool
from tools.currency_converter import CurrencyConverterTool
from tools.date_tool import DateTool
from tools.calculator_tool import CalculatorTool
from tools.stock_info_tool import StockInfoTool
from tools.stock_price_tool import StockPriceTool
from tools.stock_news_tool import StockNewsTool
from tools.stock_news_sentiment_tool import StockNewsSentimentTool
from tools.stock_dividend_tool import StockDividendTool
#from tools.plotly_tool import PlotlyPythonAstREPLTool

#chat_model = ChatOllama(model="mistral")
#tool_model = Ollama(model="mistral", temperature=temperature)

class Finn:
    def __init__(self, model_name="gpt-3.5-turbo-1106", temperature=0, max_tokens=1024, cache=True):
        self.model_name  = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.cache       = cache
        self.base_tools  = [
            "ddg-search", 
            "wikipedia",
            # "human", # i don't know how to interact with the user through chainlit
            # "llm-math", # that's not working!
        ] 
        self.finn_tools  = [
            DateTool(), 
            CalculatorTool(),
            CurrencyConverterTool(),
            StockNewsTool(),
            StockNewsSentimentTool,
            StockPriceTool(),
            StockInfoTool(),
            StockDividendTool(),
            #        PythonREPLTool(),
        ]
        
        # TODO: possible remove!
        self.optional_params = {
          "top_p": 0.8,
          "frequency_penalty": 0,
          "presence_penalty": 0
        }

    def base_llm(self, temperature=0.0):
        #return Ollama(model="mistral", temperature=self.temperature, cache=True)
        return OpenAI(
            model        = self.model_name,
            temperature  = self.temperature,
            cache        = self.cache,
#            model_kwargs = self.optional_params
        )
    
    def chat_llm(self, temperature=0.0):
        #return ChatOllama(model="mistral", temperature=self.temperature, cache=True)
        return ChatOpenAI(
            model        = self.model_name,
            temperature  = self.temperature,
            cache        = self.cache,
#            top_p        = 0.8,
#            model_kwargs = self.optional_params
        )

    @property
    def tools(self):
         all_tools = load_tools(self.base_tools, llm=self.base_llm())
         all_tools.extend(self.finn_tools)
         return all_tools
         
    @property
    def prompt(self):
        #prompt = hub.pull("hwchase17/structured-chat-agent")
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", HUMAN_PROMPT),
        ])

    @property
    def memory(self):
        # ConversationSummaryBufferMemory?
        return ConversationBufferMemory( 
            llm=self.chat_llm(),
            memory_key="chat_history",
            return_intermediate_steps=True,
            return_messages=True,
            max_token_limit=self.max_tokens)
    
    @property
    def agent(self):
        return create_structured_chat_agent(self.chat_llm(), self.tools, self.prompt)
    
    @property
    def executor(self):
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        return AgentExecutor.from_agent_and_tools(
            agent  = self.agent,
            tools  = self.tools,
            memory = self.memory,
            stop   = ["Observe:"], # , "Final Answer"TODO: Maybe with final answer
            verbose=True,
            intermediate_steps=True,
            handle_parsing_errors=True,
            agent_kwargs = {
                "memory_prompts": [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"]
            })

    @property
    def runnable(self):
        return self.executor

# BaseLLM
# https://python.langchain.com/docs/integrations/llms/ollama
    # model = ChatOllama(
    #     cache=True,
    #     callback_manager=callback_manager,
    #     model=config["model"],
    #     repeat_penalty=config["settings"]["repeat_penalty"],
    #     temperature=config["settings"]["temperature"],
    #     top_k=config["settings"]["top_k"],
    #     top_p=config["settings"]["top_p"],
    # )!