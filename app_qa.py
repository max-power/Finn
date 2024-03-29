# Code form here:
# https://docs.chainlit.io/examples/qa

# ALSO LOOK HERE:
# retrieval qa chain (summarization_chain, question_answering_chain) as tools!!!
#https://github.com/truera/trulens/blob/9db4b0c70cddd3c7ed1ff4566e928ff329e4d135/trulens_eval/examples/expositional/frameworks/langchain/langchain_retrieval_agent.ipynb


# Unbeding ausprobieren!!!!!
# https://github.com/Chainlit/cookbook/blob/main/chroma-qa-chat/app.py


# https://python.langchain.com/docs/integrations/document_loaders/news
# from langchain_community.document_loaders import NewsURLLoader
# from langchain_community.document_loaders import DataFrameLoader
# loader = DataFrameLoader(df, page_content_column="Team")
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://www.espn.com/")
# from langchain_community.document_loaders import PyPDFLoader
# loader_pdf = PyPDFLoader("../MachineLearning-Lecture01.pdf")
# from langchain_community.document_loaders.merge import MergedDataLoader
# loader_all = MergedDataLoader(loaders=[loader_web, loader_pdf])

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
import os


text_splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings      = OpenAIEmbeddings()
message_history = ChatMessageHistory()
memory          = ConversationBufferMemory(
                        memory_key="chat_history",
                        output_key="answer",
                        chat_memory=message_history,
                        return_messages=True,
                    )

from utils.file_loader import FileLoader
from chainlit.element import Element

@cl.step(name="Upload", type="run", root=False)
async def process_files(files: List[Element]):
   docs = [FileLoader(file).content for file in files]
   return Chroma.from_documents(docs, OpenAIEmbeddings())


@cl.on_chat_start
async def on_chat_start():
    files = None
    
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain", "text/csv", "application/pdf"],
            max_size_mb=20,
            timeout=180,
    ).send()

    msg = cl.Message(content="", disable_feedback=False)
    await msg.send()
    
    if files:
        msg.content = "Processing files ..."
        await msg.update()

        docsearch = await process_files(files)


    # # Process uplpaded Files
    # for file in files:
    #     msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    #     await msg.send()
    
    #     # read the uploaded file
    #     with open(file.path, "r", encoding="utf-8") as f:
    #         text = f.read()

    #     # Split the text into chunks
    #     texts = text_splitter.split_text(text)

    #     # Create a metadata for each chunk
    #     metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    #     docsearch = await cl.make_async(Chroma.from_texts)(
    #         texts, embeddings, metadatas=metadatas
    #     )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing done. You can now ask questions!"
    await msg.update()

    # save chain to session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    # read chain from session
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"Source {source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()