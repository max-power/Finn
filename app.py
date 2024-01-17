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

# LLM Cache
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path="db/langchain_llm_cache.db"))

#
from typing import Dict, Any, List
from chainlit.element import Element

OPENAI_MODELS = [
    "gpt-4-1106-preview", #The latest GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. This preview model is not yet suited for production traffic. 
    "gpt-4-vision-preview", #Ability to understand images, in addition to all other GPT-4 Turbo capabilties. Returns a maximum of 4,096 output tokens. This is a preview model version and not suited yet for production traffic
    "gpt-4", # Currently points to gpt-4-0613
    "gpt-4-32k",
    "gpt-3.5-turbo-1106", # The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens.
    "gpt-3.5-turbo", # 
    "gtp-3.5-instruct", # Similar capabilities as GPT-3 era models. Compatible with legacy Completions endpoint and not Chat Completions.
    "babbage-002", #	Replacement for the GPT-3 ada and babbage base models.
    "davinci-002", #Replacement for the GPT-3 curie and davinci base models.
]


#model = Ollama(model="mistral")
# add_llm_provider(LangchainGenericProvider(
#     id=model._llm_type,
#     name="Mistral",
#     llm=model,
#     is_chat=True,
# ))
# add_llm_provider(OpenAI)
# add_llm_provider(ChatOpenAI)



################################################################################
# Change name of Message Author in Chainlit UI
@cl.author_rename
def rename(orig_author: str):
    mapping = {
        "LLMMathChain": "Albert Einstein", 
        "Chatbot": "Assistant",
    }
    return mapping.get(orig_author, orig_author)

################################################################################
# Update settings from Settings Panel
# https://github.com/Chainlit/cookbook/blob/main/image-gen/app.py 
@cl.on_settings_update
async def setup_agent(settings):
    finn = Finn(
        model_name  = settings["model"],
        temperature = settings["temperature"],
        max_tokens  = int(settings["max_token"]),
        cache       = settings["cache"]
    )
    await cl.Message(content=f"Updated Model using {finn.model_name}").send()
    cl.user_session.set("finn", finn)
    cl.user_session.set("runnable", finn.runnable)
        
################################################################################
# Start chat
@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings([
       Select(
           id            = "model",
           label         = "OpenAI Model",
           items         = {m: m for m in OPENAI_MODELS},
           initial_value ="gpt-3.5-turbo-1106",
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
           initial       = 0.0,
       ),
       Slider(
           id            = "max_token",
           label         = "Max output tokens",
           initial       = 1024,
           min           = 0,
           max           = 2048,
           step          = 64,
       ),
       # Slider(id="top_p",label="Top P",initial=0.7,min=0,max=1,step=0.1),
    ]).send()
    await setup_agent(settings)


################################################################################
# Message handler
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Final", "Answer"]
    )])

    # check incoming message for attached files
    # files_names = await process_files(message.elements)
    # if file_names:
    #     msg.elements.append(
    #         cl.Text(name="Files", content=file_names, display="inline")
    #     )
    


    # setup response
    msg = cl.Message(content="")
    await msg.send()

    
    # send response
    response = await runnable.ainvoke({"input": message.content}, config=runnable_config)
    await msg.stream_token(response["output"])
    await msg.send()
    
    # async with cl.Step(type="run", name="QA Assistant"):
    #     async for chunk in runnable.astream({"input": message.content}, config=runnable_config):
    #         await msg.stream_token(chunk['output'])

    #await msg.send()
    
    # LOOOK HERE FOR STEP
    #https://github.com/Chainlit/cookbook/blob/aa71a1808f0edfbb6d798df90ac2467636086506/bigquery/app.py#L41
    # EVEN BETTER
    #https://github.com/Chainlit/cookbook/blob/aa71a1808f0edfbb6d798df90ac2467636086506/openai-functions/app.py#L76
    # AND HERE FOR USE WITH TOOLS AND FILES!
    # https://github.com/Chainlit/cookbook/blob/aa71a1808f0edfbb6d798df90ac2467636086506/openai-assistant/app.py#L194

################################################################################
# Process uploaded files
async def process_files(files: List[Element]):
    if not files:
        return []
        
    file_names = ", ".join([file.name for file in files])
    #file_ids = await upload_files(files)
        
    return file_names

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