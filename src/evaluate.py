"""LangSmith evaluation for the agentic RAG graph.

Four evaluators from the LangSmith RAG tutorial:
https://docs.langchain.com/langsmith/evaluate-rag-tutorial

1. correctness — generated answer vs reference answer
2. relevance — generated answer vs the user question
3. groundedness — generated answer vs retrieved documents
4. retrieval_relevance — retrieved documents vs the user question
"""

from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langsmith import Client, traceable

from graph import app
from query import LLM_MODEL

load_dotenv()

DATASET_NAME = "RAG Agent Docs Q&A"
EXPERIMENT_PREFIX = "rag-agent-eval"

# Intended graph routes are notes for us, not LangSmith fields.
# retrieve → clear doc questions
# direct_answer → greetings / math / general knowledge (router says "direct")
# retrieve then maybe rewrite → vague or misspelled doc questions

# We will use this dataset of question and answers to evaluate the RAG agent. The field keys in the dict are the input and output keys for the RAG agent. And should conventionally be called "inputs" and "outputs".
EXAMPLES = [
    {
        "inputs": {"question": "Who created Python and when was it first released?"},
        "outputs": {
            "answer": "Python was created by Guido van Rossum and first released in 1991."
        },
    },
    {
        "inputs": {"question": "What programming paradigms does Python support?"},
        "outputs": {
            "answer": "Python supports procedural, object-oriented, and functional programming."
        },
    },
    {
        "inputs": {"question": "What is pip and where are packages installed from?"},
        "outputs": {
            "answer": "pip is the standard Python package manager. Packages are typically installed from the Python Package Index (PyPI)."
        },
    },
    {
        "inputs": {"question": "Who wrote the book How It Works?"},
        "outputs": {"answer": "How It Works was written by Archibald Williams."},
    },
    {
        "inputs": {"question": "Who wrote Astronomy for Amateurs?"},
        "outputs": {
            "answer": "Astronomy for Amateurs was written by Camille Flammarion."
        },
    },
    {
        "inputs": {"question": "Hello, how are you today?"},
        "outputs": {
            "answer": "A short friendly greeting. No document facts are required."
        },
    },
    {
        "inputs": {"question": "Thanks, that's all I needed. Goodbye!"},
        "outputs": {
            "answer": "A brief polite farewell. No document facts are required."
        },
    },
    {
        "inputs": {"question": "What is 2 + 2?"},
        "outputs": {"answer": "4"},
    },
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris"},
    },
    {
        "inputs": {"question": "Who won the last FIFA World Cup?"},
        "outputs": {
            "answer": "Argentina won the 2022 FIFA World Cup. This is general knowledge, not from the local docs."
        },
    },
    {
        "inputs": {"question": "pythn gido who made it??"},
        "outputs": {
            "answer": "Python was created by Guido van Rossum."
        },
    },
    {
        "inputs": {"question": "that old book about steam and electricity, who wrote it?"},
        "outputs": {"answer": "How It Works was written by Archibald Williams."},
    },
]

# We will use this LLM to grade the RAG agent. LLM-as-a-judge.
def grader_llm(schema: type):
    return ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(schema)


# Leading underscore = helper for this file only, not meant to be imported. You can technically import it but you shouldn't.
def _flag(grade, key: str) -> bool:
    if isinstance(grade, dict):
        return bool(grade[key])
    return bool(getattr(grade, key))


# Same convention: internal helper that pulls page_content out of retrieved docs.
def _documents(outputs: dict) -> list:
    return [doc.page_content for doc in (outputs.get("documents") or [])]


@traceable(name="rag_agent")
def rag_bot(question: str) -> dict:
    # invoke() blocks the thread until END, then returns the final graph state (a dict).
    # result is the final graph state (a dict), returned by the graph app.
    result = app.invoke(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
        }
    )
    # _score is unused; underscore means "we only need the chunk, ignore the number".
    documents = [chunk for chunk, _score in (result.get("results") or [])]

    # The return dict is the output of the RAG agent. And will show up in the trace of a particular agent request for the question asked to the RAG agent.
    return {
        "answer": result.get("answer") or "",
        "documents": documents,
        "route": result.get("route"),
        "search_query": result.get("search_query"),
    }


# LangSmith calls this once per dataset row. `inputs` is that row's question dict.
def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])


# Schema the judge LLM must fill in. `...` is a placeholder; the string is the field description.
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"] # Although in this case, we're not using the explanation field, we're still defining it for the sake of completeness.
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]


correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

correctness_llm = grader_llm(CorrectnessGrade)

# This is the function that LangSmith calls to evaluate the RAG agent.
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Correctness: response vs reference answer."""
    # One multiline string for this example only — not a list of all questions.

    # This is the input to the LLM. When evaluating the RAG agent, client will pass rows of the dataset one by one. So inputs will be the question and reference_outputs will be the ground truth answer. And outputs will be the student answer.
    # Technically we've defined "outputs" as the reference answer but after executing the RAG agent, it modifies the field name to "reference_outputs" while keeping the actual output of the RAG agent in the "outputs" field. But this won't change the actual dataset we're using, it will just update it for the running evaluation.
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    grade = correctness_llm.invoke(
        [
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )

    # _flag is a helper function that returns True if the grade is correct, False otherwise.
    # It is used to return the grade to LangSmith. And will be used to generate stats of the evaluation and can be viewed graphically in the LangSmith UI.
    return _flag(grade, "correct")


class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

relevance_llm = grader_llm(RelevanceGrade)


def relevance(inputs: dict, outputs: dict) -> bool:
    """Relevance: response vs input question."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke(
        [
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return _flag(grade, "relevant")


class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

grounded_llm = grader_llm(GroundedGrade)


def groundedness(inputs: dict, outputs: dict) -> bool:
    """Groundedness: response vs retrieved documents."""
    doc_string = "\n\n".join(_documents(outputs)) or "(no documents retrieved)"
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke(
        [
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return _flag(grade, "grounded")


class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]


retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

retrieval_relevance_llm = grader_llm(RetrievalRelevanceGrade)


def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """Retrieval relevance: retrieved documents vs input question."""
    doc_string = "\n\n".join(_documents(outputs)) or "(no documents retrieved)"
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    grade = retrieval_relevance_llm.invoke(
        [
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return _flag(grade, "relevant")

# This is the function that LangSmith calls to create the dataset if it doesn't exist.
def ensure_dataset(client: Client):
    # If the dataset doesn't exist, create it.
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
        print(f"Created dataset: {DATASET_NAME} ({len(EXAMPLES)} examples)")
        return DATASET_NAME

    # If the dataset exists, read it.
    dataset = client.read_dataset(dataset_name=DATASET_NAME)

    # Get the existing examples in the dataset.
    existing = {
        example.inputs.get("question")
        for example in client.list_examples(dataset_id=dataset.id)
    }

    # Get the missing examples in the dataset.
    missing = [
        example
        for example in EXAMPLES
        if example["inputs"]["question"] not in existing
    ]

    # If there are missing examples, add them to the dataset.
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)
        print(f"Added {len(missing)} new example(s) to {DATASET_NAME}")
    else:
        print(f"Using existing dataset: {DATASET_NAME}")
    return DATASET_NAME


def write_flowchart() -> Path:
    """Write a Mermaid diagram of the compiled LangGraph."""
    mermaid = app.get_graph().draw_mermaid()
    out = Path(__file__).resolve().parent.parent / "docs" / "graph-flowchart.md"
    out.write_text(
        "# LangGraph flowchart\n\n"
        "Generated from `app.get_graph().draw_mermaid()`.\n\n"
        f"```mermaid\n{mermaid}\n```\n",
        encoding="utf-8",
    )
    print(f"Wrote flowchart to {out}")
    return out


def print_overall_stats(experiment_results) -> None:
    """Average True/False scores across all questions (pass rate per metric)."""
    totals: dict[str, list[float]] = {}
    for row in experiment_results:
        for result in row["evaluation_results"].get("results", []):
            if result.score is None:
                continue
            totals.setdefault(result.key, []).append(float(result.score))

    print("\nOverall stats (fraction True across the dataset):")
    if not totals:
        print("  No scores found.")
        return
    for key, scores in totals.items():
        rate = sum(scores) / len(scores)
        print(f"  {key}: {rate:.2%}  ({int(sum(scores))}/{len(scores)})")
    if experiment_results.url:
        print(f"\nLangSmith experiment: {experiment_results.url}")


def main():
    write_flowchart()
    # Client talks to the LangSmith API (datasets, traces, experiment scores).
    client = Client()
    ensure_dataset(client)
    # evaluate() loops the dataset: each row → target() → all 4 evaluators.

    # This will call the target function for each row in the dataset, and then call the evaluators for each row.
    # With target function, it will get the answer from the RAG agent, and then call the evaluators for each row to evaluate the answer.
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix=EXPERIMENT_PREFIX,
        metadata={
            "version": "agentic RAG: router, retrieve, grade, rewrite, generate",
            "llm": LLM_MODEL,
        },
    )
    print_overall_stats(experiment_results)
    return experiment_results


if __name__ == "__main__":
    main()
