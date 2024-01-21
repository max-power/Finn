from chainlit.server import app
from fastapi import Request
from fastapi.responses import HTMLResponse

import chainlit as cl
from finn import Finn
from langchain.schema.runnable.config import RunnableConfig
# Chainlit Chat Setting
from chainlit.playground.config import add_llm_provider, get_llm_providers
from chainlit.playground.providers.langchain import LangchainGenericProvider
from chainlit.playground.providers.openai import OpenAI, ChatOpenAI
from chainlit.input_widget import Select, Slider, Switch
from langchain_openai import OpenAIEmbeddings
# Retrievers
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Vector Store
from langchain_community.vectorstores import Chroma
# LLM Cache
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path="db/app_llm_cache.db"))
#
from typing import Dict, Any, List
from chainlit.element import Element

# List of OpenAI model names
from utils.openai_models import OPENAI_MODELS
from utils.file_loader import FileLoader

from finn_callback_handler import FinnCallbackHandler

SETTINGS_INPUTS = [
       Select(
           id            = "model",
           label         = "OpenAI Model",
           items         = {m: m for m in OPENAI_MODELS},
           initial_value = "gpt-3.5-turbo-1106",
       ),
       Switch(
           id            = "cache", 
           label         = "Caching", 
           description   = "Cache the OpenAI results.",
           initial       = True
       ),
       Slider(
           id            = "temperature",
           label         = "Temperature",
           descriptio    = "The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.0)",
           min           = 0,
           max           = 2,
           step          = 0.1,
           initial       = 0.8,
       ),
       Slider(
           id            = "max_token",
           label         = "Max output tokens",
           initial       = 1024,
           min           = 0,
           max           = 2048,
           step          = 64,
       ),
    ]

################################################################################
# Change name of Message Author in Chainlit UI
@cl.author_rename
def rename(author: str):
    mapping = {
        "LLMMathChain": "Calculator", 
        "LLMChain": "Assistant",
        "RunnableAssign<agent_scratchpad>": "Agent",
        "AgentExecutor": "Finn",
        "ChatOpenAI": "LLM"

    }
    return mapping.get(author, author)


################################################################################
# Update settings from Settings Panel
# https://github.com/Chainlit/cookbook/blob/main/image-gen/app.py 
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
@cl.on_settings_update
async def setup_agent(settings):
    finn = Finn(
        model_name  = settings["model"],
        temperature = settings["temperature"],
        max_tokens  = int(settings["max_token"]),
        cache       = settings["cache"]
    )
    cl.user_session.set("finn", finn)
    cl.user_session.set("runnable", finn.runnable)



################################################################################
# Start chat
@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(SETTINGS_INPUTS).send()
    await setup_agent(settings)

################################################################################
# Message handler
@cl.on_message
async def on_message(message: cl.Message):
    finn = cl.user_session.get("finn")
    runnable = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[
        cl.AsyncLangchainCallbackHandler(),
        #FinnCallbackHandler()
    ])

    # setup response
    msg = cl.Message(content="")
    await msg.send()

    # check incoming message for attached files
    #if message.elements:
    #    files = await process_files(message.elements)
    #docs = [FileLoader(file).content for file in files]
    #docs_ids = await finn.vectorstore.aadd_documents(docs)

    #response = await finn.answer(message.content, config=runnable_config)
    # generate response
    #response = await runnable.ainvoke({"input": message.content}, config=runnable_config)
    #await msg.send()

    #response = await runnable.ainvoke({"input": message.content}, config=runnable_config)
    # send respons
#    msg.content=response["output"]
    #await msg.send()

    async with cl.Step(type="llm", name="Finn", root=True):
        async for chunk in runnable.astream({"input": message.content}, config=runnable_config):
            #print('Current Step:', cl.context.current_step, '############################')
            if chunk.get("output"):
                #msg.content=chunk.get("output")
                #await msg.update()
                figure=cl.user_session.get("figure")
                if figure:
                    msg.elements.append(cl.Plotly(name="chart", figure=figure, display="inline"))
                
                await msg.stream_token(chunk.get("output"))


        
        #await msg.update()

    await msg.send()
    # LOOOK HERE FOR STEP
    #https://github.com/Chainlit/cookbook/blob/aa71a1808f0edfbb6d798df90ac2467636086506/bigquery/app.py#L41
    # EVEN BETTER
    #https://github.com/Chainlit/cookbook/blob/aa71a1808f0edfbb6d798df90ac2467636086506/openai-functions/app.py#L76
    # AND HERE FOR USE WITH TOOLS AND FILES!
    # https://github.com/Chainlit/cookbook/blob/aa71a1808f0edfbb6d798df90ac2467636086506/openai-assistant/app.py#L194



################################################################################
# Stop the current task
@cl.on_stop
def on_stop():
    runnable = cl.user_session.get("runnable")
    # TODO: That's a Hack!
    #runnable.max_iterations = 0
    print("The user wants to stop the task!")
#    runnable = cl.user_session.get("runnable")


################################################################################
# Process uploaded files
@cl.step(name="Upload", type="run", root=False)
async def process_files(files: List[Element]):
    docs = [FileLoader(file).content for file in files]
    return Chroma.from_documents(docs, OpenAIEmbeddings())

################################################################################
# Other Pages
@app.get("/hello")
def hello(request: Request):
    print(request.headers)
    return HTMLResponse("Hello World")

################################################################################
# https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html
# - abatch(inputs[, config, return_exceptions])
#       Default implementation runs ainvoke in parallel using asyncio.gather.
# - ainvoke(input[, config])
#       Default implementation of ainvoke, calls invoke from a thread.
# - assign(**kwargs)
#       Assigns new fields to the dict output of this runnable.
# - astream(input[, config])
#       Default implementation of astream, which calls ainvoke.
# - astream_log()
#       Stream all output from a runnable, as reported to the callback system.
# - atransform(input[, config])
#       Default implementation of atransform, which buffers input and calls astream.
# - batch(inputs[, config, return_exceptions])
#       Default implementation runs invoke in parallel using a thread pool executor.
# - bind(**kwargs)
#       Bind arguments to a Runnable, returning a new Runnable.

#model = Ollama(model="mistral")
# add_llm_provider(LangchainGenericProvider(
#     id=model._llm_type,
#     name="Mistral",
#     llm=model,
#     is_chat=True,
# ))
# add_llm_provider(OpenAI)
# add_llm_provider(ChatOpenAI)