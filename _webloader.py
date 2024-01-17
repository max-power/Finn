from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader



def read_url(url: str):
    return WebBaseLoader(uri).load()

uri  = "https://python.langchain.com/cookbook"
data = read_url(url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)





# https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#
# HTMLHeaderTextSplitter
#     .split_text_from_url
#     .split_text_from_file
#     .split_text
#
# RecursiveCharacterTextSplitter
#     split_text
#
# NLTKTextSplitter
# SpacyTextSplitter
# PythonCodeTextSplitter
# MarkdownTextSplitter
# LatexTextSplitter
