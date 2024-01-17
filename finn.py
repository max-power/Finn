# When asked about company news, you will also provide a sentiment using the Stock-News-Sentiment-Tool. 
SYSTEM_PROMPT = '''
You're the most seasoned financial analyst and investment advisor with expertise in stock market analysis and investment strategies. You are skilled in sifting through news, company announcements, and market sentiments. You combine various analytical insights to formulate strategic investment advice. To ensure the accuracy of temporal information, you will access the current date using a tool and utilize it to retrieve the latest data. You have access to yahoo finance data and company information using the appropiate tools. You also can conduct web searches and refer to Wikipedia for comprehensive information. You will ensure to include the sentiment analysis when providing news headlines.

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
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
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
{input}\n\n{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)
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
    openai_model = "gpt-3.5-turbo-1106"
    base_tools   = ["human", "ddg-search", "wikipedia"] # "llm-math" # that's not working!
    finn_tools   = [
        DateTool(), 
        CalculatorTool(),
        CurrencyConverterTool(), # seem to be offline
        StockNewsTool(),
        StockNewsSentimentTool,
        StockPriceTool(),
        StockInfoTool(),
        StockDividendTool(),
#        PythonREPLTool(),
    ]

    def tool_llm(self, temperature=0.0):
        #return Ollama(model="mistral", temperature=temperature, cache=True)
        return OpenAI(model=self.openai_model, temperature=temperature, cache=True)
    
    def chat_llm(self, temperature=0.0):
        #return ChatOllama(model="mistral", temperature=temperature, cache=True)
        return ChatOpenAI(model=self.openai_model, temperature=temperature, cache=True)

    @property
    def agent(self):
        return AgentFactory(chat_llm=self.chat_llm(), tools=self.tools).agent()
    
    @property
    def tools(self):
        t = load_tools(self.base_tools, llm=self.tool_llm())
        t.extend(self.finn_tools)
        return t
        
    

#BaseLLM
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

from langchain_core.language_models.chat_models import BaseChatModel
# https://github.com/DevGauge/LangChainAgentFactory
class AgentFactory:
    def __init__(self, chat_llm: BaseChatModel, tools: list[Tool] = [], memory = None, max_tokens=2000):
        """generates agents for the user to interact with the LLM.

        Args:
            tools (list[Tool], optional): Tools/StructuredTools the agent will have access to. Tools cannot be passed in to an agent after calling `agent` 
            to create an agent with different tooling, change the properties and call `agent` again. Defaults to [].

            openai_model_name (str, optional): the openai chat model to use. Defaults to 'gpt-4'.

            temperature (float, optional): how "creative" or determinative the model is. Defaults to 0.0.

            memory (_type_, optional): type of memory the agent will have. Defaults to ConversationSummaryBufferMemory which holds a summary of
            the conversation up to `max_tokens`.

            max_tokens (int, optional): The maximum number of tokens before the ConversationBufferMemory resets. Defaults to 2000.
        """
        self.llm    = chat_llm
        self.memory = memory or ConversationBufferMemory( # ConversationSummaryBufferMemory
                        llm=self.llm,
                        memory_key="chat_history",
                        return_intermediate_steps=True,
                        return_messages=True,
                        max_token_limit=max_tokens)
        self.tools  = tools
        #self.prompt = hub.pull("hwchase17/structured-chat-agent")
        self.prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", HUMAN_PROMPT),
        ])


    def agent(self, verbose=True):
        """generate an agent given self.tools, self.llm, self.memory, and agent_type

        Args:
            agent_type (_type_, optional): The type of agent that will be generated. 
            Defaults to AgentType.OPENAI_MULTI_FUNCTIONS to allow for multi-parameter functions.

            verbose (bool, optional): Will console output be verbose? Defaults to True.

        Returns:
            Agent: A chatbot with access to functions/tools and a memory.
        """
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        agent        = create_structured_chat_agent(self.llm, self.tools, self.prompt)
        agent_chain  = AgentExecutor.from_agent_and_tools(
            agent  = agent,
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

        return agent_chain

