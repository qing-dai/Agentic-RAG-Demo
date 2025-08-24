from langchain.schema import Document
from ..services.retrieve_docs import retrieve_docs



def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieval
    documents = retrieve_docs(question)
    return {"documents": documents, "question": question}