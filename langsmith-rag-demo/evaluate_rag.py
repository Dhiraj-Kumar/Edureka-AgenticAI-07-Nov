from langsmith import Client, evaluate
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from rag import rag_pipeline
import json
import re

load_dotenv()

judge = ChatOpenRouter(model="gpt-4.1-mini")
client = Client()
# This is the function for a judge to assign score for each evaluation


def grade(criteria: str, payload: str) -> dict:
    raw = judge.invoke(
        f'{criteria}\n\nReply with ONLY JSON: {{"score": 0 or 1, "reasoning": "one sentence"}}'
        f"\n\n{payload}"
    ).content
    try:
        parsed = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        return {"score": int(parsed["score"]), "comment": parsed["reasoning"]}
    except Exception:
        return {"score": 0, "comment": f"Unparseable judge output: {raw[:150]}"}

# Evaluators


def correctness(inputs, outputs, reference_outputs):
    return {"key": "correctness", **grade(
        "Score 1 if the student answer conveys the same facts as the reference answer "
        "(different wording is fine). Score 0 if a fact contradicts it or is missing.",
        f"QUESTION: {inputs['question']}\n"
        f"REFERENCE: {reference_outputs['answer']}\n"
        f"STUDENT: {outputs['answer']}",
    )}


def groundedness(inputs, outputs):
    return {"key": "groundedness", **grade(
        "Score 1 if every claim in the answer is supported by the retrieved context. "
        "Score 0 if the answer states anything absent from the context.",
        f"CONTEXT:\n{outputs['context']}\n\nANSWER: {outputs['answer']}",
    )}


def retrieval_recall(inputs, outputs, reference_outputs):
    return {"key": "retrieval_recall", **grade(
        "You are judging the retriever, not the answer. Score 1 if the context contains "
        "the information needed to produce the reference answer, else 0.",
        f"QUESTION: {inputs['question']}\n"
        f"REFERENCE: {reference_outputs['answer']}\n"
        f"CONTEXT:\n{outputs['context']}",
    )}


results = evaluate(
    rag_pipeline,
    data="hr-policy-rag-eval-data",
    evaluators=[correctness, groundedness, retrieval_recall],
    experiment_prefix="hr-rag-openrouter",
    max_concurrency=4,
    client=client,
    metadata={"chunk_size": 500, "k": 5, "llm": "gpt-4.1-mini"}
)

print(results)
