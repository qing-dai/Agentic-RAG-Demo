from langgraph.graph import END, StateGraph, START
from agent import app
from agent.app.graph.graph_chain import GraphState
from agent.app.nodes.evaluate_query import query_evaluate
from pprint import pprint
from agent.app.nodes.extract_state import extract_ticker
from agent.app.nodes.yahoo_finance_state import yahoo_search
from agent.app.nodes.retrieve import retrieve
from agent.app.nodes.evaluate_documents import evaluate_documents
from agent.app.nodes.generate import generate
from IPython.display import Image, display


def build_app():
    workflow = StateGraph(GraphState)
    # Define the nodes
    # retrieve {"documents": documents, "question": question}
    workflow.add_node("retrieve", retrieve)  

    # evaluate documents {"documents": filtered_docs, "question": question, "web_search": web_search}
    workflow.add_node("evaluate_documents", evaluate_documents)  

    # generate {"documents": documents, "question": question, "generation": generation}
    workflow.add_node("generate", generate) 

    # web search {"documents": documents, "question": question}
    workflow.add_node("extract_ticker", extract_ticker)  

    workflow.add_node("yahoo_search", yahoo_search)


    # Build graph
    workflow.add_conditional_edges(
                        START, 
                        query_evaluate,
                        {
                            "IS ABOUT TICKER": "extract_ticker",
                            "IS NOT ABOUT TICKER": "retrieve",
                        },)
    workflow.add_edge("retrieve", "evaluate_documents")
    workflow.add_edge("evaluate_documents", "generate")
    workflow.add_edge("extract_ticker", "yahoo_search")
    workflow.add_edge("yahoo_search", "generate")
    workflow.add_edge("generate", END)
    # Compile
    return workflow.compile()



# Save the graph image locally
# png_bytes = app.get_graph(xray=True).draw_mermaid_png()

# output_path = "workflow_graph.png"
# with open(output_path, "wb") as f:
#     f.write(png_bytes)

# print(f"Graph saved to {output_path}")