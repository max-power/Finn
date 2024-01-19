from chainlit import File
from mimetypes import guess_type
from langchain_community.document_loaders import (
    TextLoader, 
    PyMuPDFLoader, 
    JSONLoader, 
    CSVLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

class FileLoader:
    loaders = {
        "text/plain": TextLoader,
        "text/csv": CSVLoader,
        "application/pdf": PyMuPDFLoader,
        "application/json": JSONLoader,
        "application/msword": UnstructuredWordDocumentLoader,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": UnstructuredWordDocumentLoader,
        "application/vnd.ms-excel": UnstructuredExcelLoader,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": UnstructuredExcelLoader,
    }
    def __init__(self, file: File):
        self.file = file
        self.mime = guess_type(file.path)[0]
    
    @property
    def loader(self):
        return self.loaders.get(self.mime)
    
    @property
    def content(self):
        # load_and_split parameter: text_splitter: defaults to RecursiveCharacterTextSplitter
        return self.loader(self.file.path).load_and_split()[0] 
