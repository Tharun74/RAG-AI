from langchain_community.retrievers import ArxivRetriever


def arxiv_retriver(prompt):
    retriever = ArxivRetriever(
    load_max_docs=2,
    get_full_documents=True
    )
    
    docs = retriever.invoke(prompt)
    return docs