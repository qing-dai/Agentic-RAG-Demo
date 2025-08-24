from pprint import pprint
from agent.app.graph.build import build_app

def run_once(question: str):
    graph = build_app()
    last = {}
    for output in graph.stream({"question": question}):
        for node, state in output.items():
            print(f"Node '{node}':")
            pprint(state, indent=2, width=100)
            last = state
        print("\\n---\\n")
    print(last.get("generation"))

if __name__ == "__main__":
    run_once("What is the city of Zurich like?")


# Save the graph image locally
# png_bytes = app.get_graph(xray=True).draw_mermaid_png()

# output_path = "workflow_graph.png"
# with open(output_path, "wb") as f:
#     f.write(png_bytes)

# print(f"Graph saved to {output_path}")