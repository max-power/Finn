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
from chainlit.input_widget import Select, Slider

# LLM Cache
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

#model = Ollama(model="mistral")
# add_llm_provider(LangchainGenericProvider(
#     id=model._llm_type,
#     name="Mistral",
#     llm=model,
#     is_chat=True,
# ))
# add_llm_provider(OpenAI)
# add_llm_provider(ChatOpenAI)

# https://github.com/Chainlit/cookbook/blob/main/image-gen/app.py 
@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
           [
               Select(
                   id="Model",
                   label="OpenAI - Model",
                   values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                   initial_index=0,
               ),
               Slider(
                   id="temperature",
                   label="Temperature",
#                   initial=config["settings"]["temperature"],
                   min=0,
                   max=1,
                   step=0.1,
                   description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
               ),
           ]
       ).send()
    value = settings["Model"]
    
    
    runnable = Finn().agent
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    runnable_config = RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Final", "Answer"]
    )])

    msg = cl.Message(content="")
    await msg.send()
    
    response = await runnable.ainvoke({"input": message.content}, config=runnable_config)
    await msg.stream_token(response["output"])
    await msg.send()

@app.get("/hello")
def hello(request: Request):
    print(request.headers)
    return HTMLResponse("Hello World")

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