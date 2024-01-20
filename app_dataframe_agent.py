from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import load_agent_executor
import pandas as pd
from langchain_openai import OpenAI

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a CSV file to begin!",
            accept=["text/csv"],
            max_size_mb=20,
            timeout=180,
        ).send()

    msg = cl.Message(content="", disable_feedback=True)
    await msg.send()

    if files:
        msg.content = "Processing files ..."
        await msg.update()

        df = pd.read_csv(files[0].path)
    
    # Let the user know that the system is ready
    msg.content = f"Processing done. You can now ask questions!"
    await msg.update()

    model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
    agent = create_pandas_dataframe_agent(
        model, df, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True, include_df_in_prompt=True,
        agent_executor_kwargs={
            'handle_parsing_errors': True
        }
    )
    cl.user_session.set("chain", agent)

@cl.on_message
async def main(message: cl.Message):
    # read chain from session
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer  = True,)


    msg = cl.Message(content="")
    msg.send()

    response = await chain.acall({'input': message.content}, callbacks=[cb])
    msg.content = response["output"]

    await msg.send()





# if __name__ == "__main__":
#     df = pd.read_csv("__files_for_testing/titanic.csv")
#     model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
#     agent = create_pandas_dataframe_agent(model, df, verbose=True)
#     agent.invoke({"input": "how many rows are there?"} )
#     agent.invoke({"input": "how many rows in the age column are different?"})
#     agent.invoke({"input": "whats the square root of the average age?"})