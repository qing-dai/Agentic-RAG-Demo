from ..services.retrieval_evaluator_class import RetrievalEvaluatorService




def evaluate_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    retrieval_grader = RetrievalEvaluatorService()
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    for d in documents:
        # RetrievalEvaluatorService.evaluate(document, question) returns a structured output
        score = retrieval_grader.evaluate(d["text"], question)
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"documents": filtered_docs, "question": question}