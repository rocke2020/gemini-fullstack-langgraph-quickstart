from src.agent import graph

config = {"configurable": {"max_research_loops": 2, "initial_search_query_count": 3}}

input_message = {
    "messages": [
        {"role": "user", "content": "分析2026年的金融研投行业的AI agent发展趋势"}
    ]
}
state = graph.invoke(
    input_message,  # type: ignore
    config=config,  # type: ignore
)
print(f'{state = }')
