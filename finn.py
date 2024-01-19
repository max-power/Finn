from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain import hub
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.agents.tools import Tool
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.chains import ConversationChain, LLMMathChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ChatMessageHistory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import load_tools
from tools.tools import *

from langchain.tools.retriever import create_retriever_tool

from langchain_experimental.tools import PythonREPLTool
from tools.currency_converter import CurrencyConverterTool
from tools.date_tool import DateTool
from tools.calculator_tool import CalculatorTool
from tools.stock_toolkit import StockToolkit
from tools.human_input_tool import HumanInputChainlit
from tools.plotly_tool import PlotlyPythonAstREPLTool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

#chat_model = ChatOllama(model="mistral")
#tool_model = Ollama(model="mistral", temperature=temperature)

from datetime import datetime

from prompts.finn_prompt import *

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
            #HumanInputChainlit(),
            PlotlyPythonAstREPLTool(),
        ]
        
        # TODO: possible remove!
        self.optional_params = {
          "top_p": 0.8,
          "frequency_penalty": 0,
          "presence_penalty": 0
        }

    @property
    def base_llm(self):
        #return Ollama(model="mistral", temperature=self.temperature, cache=True)
        return OpenAI(
            model        = self.model_name,
            temperature  = self.temperature,
            cache        = self.cache,
#            model_kwargs = self.optional_params
        )
    @property
    def chat_llm(self):
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
        all_tools = load_tools(self.base_tools, llm=self.base_llm)
        all_tools.extend(self.finn_tools)
        all_tools.extend(StockToolkit().get_tools())
        all_tools.append(self.retriever_tool)
        return all_tools
         
    @property
    def prompt(self):
        #return hub.pull("hwchase17/structured-chat-agent")
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("system", f"Today is {datetime.now().strftime('%A, %B %d, %Y %H:%M:%S')}."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", HUMAN_PROMPT),
        ])

    @property
    def memory(self):
        # ConversationBufferMemory
        return ConversationSummaryBufferMemory( 
            llm             = self.chat_llm,
            memory_key      = "chat_history",
            output_key      = "output",
            chat_memory     = ChatMessageHistory(),
            max_token_limit = self.max_tokens,
            return_messages = True,
            return_intermediate_steps = False,
        )
    
    @property
    def agent(self):
        return create_structured_chat_agent(self.chat_llm, self.tools, self.prompt)
    
    @property
    def executor(self):
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        return AgentExecutor.from_agent_and_tools(
            agent  = self.agent,
            tools  = self.tools,
            memory = self.memory,
            max_iterations        = 10,
            max_execution_time    = 30, # seconds
            verbose               = True,
            intermediate_steps    = True,
            handle_parsing_errors = True,
#            return_only_outputs   = True,
#            include_run_info      = True,
            agent_kwargs = {
                "memory_prompts":  [chat_history],
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            })

    @property
    def runnable(self):
        return self.executor
        
    @property
    def vectorstore(self):
        return Chroma("document_store", self.embeddings)
    
    @property
    def text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    
    @property
    def embeddings(self):
        return OpenAIEmbeddings()
    
    @property
    def vectorstore(self):
        return Chroma("document_store", self.embeddings)

    @property
    def retriever_tool(self):
        return create_retriever_tool(
            self.vectorstore.as_retriever(),
            "VectorStoreDocumentSearch",
            "Searches and returns documents from the human has uploaded. Useful when you need to answer questions about files the user uploaded.",
        )


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