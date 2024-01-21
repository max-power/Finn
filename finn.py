from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain import hub
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.agents.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_json_agent, create_react_agent, create_openai_tools_agent
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

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.tools.render import render_text_description_and_args
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

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
            #PlotlyPythonAstREPLTool(),
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
            streaming    = False,
#            model_kwargs = self.optional_params
        )
    @property
    def chat_llm(self):
        #return ChatOllama(model="mistral", temperature=self.temperature, cache=True)
        return ChatOpenAI(
            model        = self.model_name,
            temperature  = self.temperature,
            cache        = self.cache,
            streaming    = False,
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
        #return hub.pull("hwchase17/openai-tools-agent")
        #return hub.pull("hwchase17/structured-chat-agent")
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("system", f"Today is {datetime.now().strftime('%A, %B %d, %Y %H:%M:%S')}."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", HUMAN_PROMPT),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])
    
    @property
    def memory(self):
        # ConversationSummaryBufferMemory
        return ConversationBufferMemory( 
            llm             = self.chat_llm,
            memory_key      = "chat_history",
            output_key      = "output",
            chat_memory     = self.history,
            max_token_limit = self.max_tokens,
            return_messages = True,
            return_intermediate_steps = True,
        )
    
    @property
    def history(self):
        return ChatMessageHistory()

    @property
    def agent(self):
        #return create_structured_chat_agent(self.chat_llm, self.tools, self.prompt)

        #return create_openai_tools_agent(self.chat_llm, self.tools, self.prompt)
    
        missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
            self.prompt.input_variables
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        prompt = self.prompt.partial(
            tools=render_text_description_and_args(list(self.tools)),
            tool_names=", ".join([t.name for t in self.tools]),
        )
        llm_with_tools = self.chat_llm.bind(
            tools=[format_tool_to_openai_tool(tool) for tool in self.tools],
            #tool_names=", ".join([t.name for t in self.tools]),
            stop=["Observation:", "Observation: ", "<FINAL_ANSWER>"]
        )
        
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            )
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        return agent

        #structured_chat_Agent_
        # prompt = self.prompt.partial(
        #     tools=render_text_description_and_args(list(self.tools)),
        #     tool_names=", ".join([t.name for t in self.tools]),
        # )
        # llm_with_stop = self.chat_llm.bind(stop=["Observation:", "Observation: ", "<FINAL_ANSWER>"])

        # chain = (
        #     RunnablePassthrough.assign(
        #         agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        #     )
        #     | prompt
        #     | llm_with_stop
        #     | JSONAgentOutputParser()
        # )
        # return chain
    
    @property
    def executor(self):
        chat_history = MessagesPlaceholder(variable_name="chat_history")
        return AgentExecutor.from_agent_and_tools(
            agent  = self.agent,
            tools  = self.tools,
            memory = self.memory,
            max_iterations        = 10,
            #max_execution_time    = 30, # seconds
            verbose               = True,
            intermediate_steps    = True, 
            handle_parsing_errors = True, #"Something went wrong",
            return_only_outputs   = True,
            #include_run_info      = True,
            early_stopping_method = "force", # does not support "generate"
            
            agent_kwargs = {
                #"memory_prompts":  [chat_history],
                #"input_variables": ["input", "agent_scratchpad", "chat_history"],
#                "early_stopping_method": "generate",
                #"stop": ["Observation:", "\nObservation", "<FINAL_ANSWER>"],
            }
        )

    @property
    def runnable(self):
        return self.executor
            
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

    async def answer(self, input, config=None):
        return await self.runnable.ainvoke({"input": input}, config=config)



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

# if __name__ == "__main__":
#     f = Finn()
#     f.executor.invoke({'input': 'What is the current apple stock price'})