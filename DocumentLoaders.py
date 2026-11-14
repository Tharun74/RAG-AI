from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

def load_txt_file(path):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return docs

def load_pdf_file(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs
    
def load_web_page(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs
