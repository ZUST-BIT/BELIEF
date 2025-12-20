from query import query
from agentic_tools import GeneratorAgent, MultiEvaluator
from retriever import retrieve_process

if __name__ == '__main__':
    # init
    eval = MultiEvaluator()
    gener = GeneratorAgent()
    knowledge = retrieve_process(query)
    eval_knowledge = eval.evaluate(query, knowledge)
    evidence = f"expert analysis: {eval_knowledge}"
    answer = gener.generate_answer(query, evidence)
    print(answer)