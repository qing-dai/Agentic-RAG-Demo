from ..services.query_evaluator import QueryEvaluatorService

_svc: QueryEvaluatorService | None = None

def _get_service() -> QueryEvaluatorService:
    global _svc
    if _svc is None:
        _svc = QueryEvaluatorService()
    return _svc


def query_evaluate(state):
    question = state["question"]
    svc = _get_service()
    eva_res = svc.score(question)
    if eva_res == "yes":            
        print("---GRADE: QUESTION IS ABOUT TICKER PRICE---")
        return "IS ABOUT TICKER"
    else:
        print("---GRADE: QUESTION IS NOT ABOUT TICKER PRICE---")
        return "IS NOT ABOUT TICKER"
