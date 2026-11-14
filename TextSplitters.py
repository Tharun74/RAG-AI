from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter

def split_as_character(docs):
    splitter = CharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 10
    )
    chunks = splitter.split_documents(docs)
    return chunks

def split_as_tokens(docs):
    splitter = TokenTextSplitter(
        chunk_size = 200,
        chunk_overlap = 10
    )
    chunks = splitter.split_documents(docs)
    return chunks

def split_as_recursive_character(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 10,
        length_function = len,
        keep_separator=False
    )
    chunks = splitter.split_documents(docs)
    return chunks
    
    