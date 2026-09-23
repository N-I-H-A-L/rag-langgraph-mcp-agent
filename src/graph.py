import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from query import LLM_MODEL, query_chunks

# -----------------------------------------------------------------------------
# How LangGraph works (this file in one picture)
#
# A LangGraph app is a flowchart:
#
#   START → decide_to_retrieve → retrieve → grade_documents → generate → END
#                             ↘            ↘ rewrite_question ↗
#                              direct_answer → END
#
# Pieces:
#
# STATE
#   A dict that travels through the graph. Every node receives the current
#   state and may return a partial update. LangGraph merges that update into
#   the state (it does not replace the whole dict). Unused keys stay as they
#   were. Example: decide_to_retrieve only returns {"route": "..."} so
#   "question" is still there for later nodes.
#
#   Most keys use the default merge: new value REPLACES the old one
#   (question, route, answer, ...).
#
#   `messages` is different. It uses the add_messages REDUCER, which APPENDS
#   HumanMessage / AIMessage objects instead of overwriting the list. That is
#   how a chat history is built turn by turn. The list lives in this state
#   dict only. There is no checkpointer: when invoke() finishes, we keep the
#   list in a normal Python variable and pass it into the next invoke().
#
# NODE
#   One step in the flowchart. It is just a Python function:
#       def my_node(state) -> dict:
#           ...
#           return {"some_key": new_value}
#   You register it with graph.add_node("name", my_node).
#   The string "name" is what edges point at.
#
# EDGE
#   A fixed arrow: after node A, always go to node B.
#       graph.add_edge("retrieve", "generate")
#   START and END are special built-in points (not functions you write).
#   START = where invoke() begins. END = stop and return the state.
#
# CONDITIONAL EDGE
#   A fork: after node A, pick the next node from the current state.
#   You pass a function that reads state and returns a string. That string
#   is looked up in a path_map to get the real next node:
#       "retrieve"      -> retrieve node
#       "direct_answer" -> direct_answer node
#
#   After retrieve we use a SECOND conditional edge (not a normal edge):
#       "generate" -> generate   (docs look relevant)
#       "rewrite"  -> rewrite_question → retrieve again
#
# COMPILE + INVOKE
#   graph.compile() freezes the flowchart into a runnable app.
#   app.invoke({"question": "...", "messages": [...]})
#   walks the edges and returns the final state (including "answer"
#   and the updated messages list).
# -----------------------------------------------------------------------------

# RAG answer: system instructions + the stored chat history (`messages`).
# from_messages is a function that takes a list of messages, the whole conversation history, and returns a prompt template.
GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", # role
            """You are a helpful assistant. Answer the question using only the context below.
If the context does not contain the answer, say you don't know.
Use earlier messages if the user is asking a follow-up.

Context:
{context}""", # template text
        ),
        MessagesPlaceholder("messages"), # messages placeholder is a special placeholder that LangGraph uses to inject the conversation history into the prompt.
    ]
)

# No documents. Still gets the stored message history.
DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the user's question. You do not have extra documents for this one.
If you are not sure, say so. Use earlier messages if this is a follow-up.""",
        ),
        MessagesPlaceholder("messages"),
    ]
)

# Router also sees history so follow-ups like "tell me more" can still retrieve.
# This prompt is used to determine whether to retrieve or direct the answer.
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a router for a question-answering system.

The knowledge base contains documents about:
- Python programming
- Astronomy for amateurs
- How machines and everyday technology work

If the latest user question (or a follow-up about those topics) can be answered
from those documents, reply with exactly: retrieve
Otherwise (greetings, small talk, unrelated topics), reply with exactly: direct""",
        ),
        MessagesPlaceholder("messages"),
    ]
)

# After retrieve: are these chunks actually useful for the user's question?
GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You grade retrieved documents for a RAG system.

Question:
{question}

Documents:
{context}

If the documents contain information that can help answer the question, reply with exactly: relevant
If they are off-topic or unhelpful, reply with exactly: not_relevant
"""
)

# Used when the grader says the docs are not relevant.
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the search query so it is more likely to match documents about
Python, amateur astronomy, or how machines / everyday technology work.

The rewritten query should be a single search-style question, not an answer.
Keep the same intent as the original question.

Original question:
{question}

Previous search query:
{search_query}

Reply with only the rewritten query.
"""
)

MAX_REWRITES = 2


class RAGState(TypedDict):
    # TypedDict = "this dict should have these keys and types". It is like a schema or blueprint for the state. That these keys and types are required for the state. It is a way to enforce the structure of the state.

    # StateGraph(RAGState) uses it as the schema for the shared state.
    question: str  # this turn's user text; default merge REPLACES each invoke

    route: Literal["retrieve", "direct_answer"]  # written by the router node
    # Literal = "this key should be one of these values". Similar to enum in other languages.

    search_query: str  # text sent to Chroma; starts as the user question, may be rewritten

    grade: Literal["generate", "rewrite"]  # written by grade_documents

    rewrite_count: int  # how many times we have already rewritten; stops infinite loops

    context: str  # retrieved chunk text, used by generate

    results: list  # (Document, score) pairs from Chroma; empty on the direct path

    answer: str  # final reply, written by generate or direct_answer

    # Annotated = "this key should have this type".
    # add_messages = APPEND Human/AI messages instead of replacing the list.
    # add_messages is a reducer that appends Human/AI messages instead of replacing the list. It is a special function that LangGraph uses to append messages to the list so we don't have to do it manually.
    messages: Annotated[list[AnyMessage], add_messages]


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(model=LLM_MODEL, temperature=0)


def decide_to_retrieve(state: RAGState) -> dict:
    # NODE: runs first. Reads messages (includes this turn's HumanMessage),
    # writes state["route"]. Does not touch `messages` itself.
    # We're not appending this node's messages to the list since it's an internal vote whether to retrieve or direct the answer and is not relevant to the conversation history.
    decision = (ROUTER_PROMPT | get_llm()).invoke({"messages": state["messages"]})
    text = decision.content.strip().lower()
    route = "retrieve" if "retrieve" in text else "direct_answer"
    print(f"Router chose: {route}")
    return {"route": route}
    # When returned, the "route" key is added to the state by the StateGraph.


def retrieve(state: RAGState) -> dict:
    # NODE: only runs if the router (or a rewrite loop) sent us here.
    # Uses search_query when present so a rewritten question can be tried.
    search_query = state.get("search_query") or state["question"]
    results = query_chunks(search_query)  # Chroma similarity search
    context = "\n\n".join(chunk.page_content for chunk, _score in results)
    print(f"Retrieved {len(results)} chunk(s) for: {search_query}")
    return {
        "results": results,
        "context": context,
        "search_query": search_query,
        "rewrite_count": state.get("rewrite_count") or 0,
    }
    # Note: No appending of messages. LangGraph merges these keys into state.


def generate(state: RAGState) -> dict:
    # NODE: RAG answer. Needs context that retrieve put into state.
    # Returning an AIMessage is APPENDED to state["messages"] by add_messages.
    response = (GENERATE_PROMPT | get_llm()).invoke(
        {"context": state["context"], "messages": state["messages"]}
    )
    return {"answer": response.content, "messages": [AIMessage(content=response.content)]}
    # When "messages" key is returned, it is appended to the list by add_messages.


def direct_answer(state: RAGState) -> dict:
    # NODE: other branch of the fork. No Chroma call. Clears results so
    # the print loop at the bottom has nothing to show as sources.
    response = (DIRECT_PROMPT | get_llm()).invoke({"messages": state["messages"]})
    return {
        "answer": response.content,
        "results": [],
        "context": "",
        "messages": [AIMessage(content=response.content)],
    }


def grade_documents(state: RAGState) -> dict:
    # NODE: after retrieve. Asks the LLM if the chunks match the user question.
    # Writes `grade` so a CONDITIONAL EDGE can choose generate vs rewrite.
    decision = (GRADE_PROMPT | get_llm()).invoke(
        {"question": state["question"], "context": state["context"]}
    )
    text = decision.content.strip().lower()
    # Check not_relevant first: it contains the substring "relevant".
    if "not_relevant" in text:
        grade = "rewrite"
    elif "relevant" in text:
        grade = "generate"
    else:
        grade = "rewrite"
    print(f"Grader chose: {grade}")
    return {"grade": grade}


def rewrite_question(state: RAGState) -> dict:
    # NODE: docs were not relevant. Rewrite the Chroma query and try retrieve again.
    rewritten = (REWRITE_PROMPT | get_llm()).invoke(
        {
            "question": state["question"],
            "search_query": state.get("search_query") or state["question"],
        }
    )
    new_query = rewritten.content.strip()
    rewrite_count = (state.get("rewrite_count") or 0) + 1
    print(f"Rewrote search query (attempt {rewrite_count}): {new_query}")
    return {"search_query": new_query, "rewrite_count": rewrite_count}


def choose_branch(state: RAGState) -> str:
    # Not a node. CONDITIONAL EDGE after decide_to_retrieve.
    # Returns a node name; does not merge anything into state.
    return state["route"]


def choose_after_grade(state: RAGState) -> str:
    # Not a node. CONDITIONAL EDGE after grade_documents.
    # If we have rewritten too many times, stop looping and generate anyway.
    if (state.get("rewrite_count") or 0) >= MAX_REWRITES and state.get("grade") == "rewrite":
        print("Max rewrites reached; generating with the current documents.")
        return "generate"
    return state["grade"]



# 1) Create an empty graph that uses RAGState as its memory.
graph = StateGraph(RAGState)

# 2) Register NODES: name (string) → function to run.
graph.add_node("decide_to_retrieve", decide_to_retrieve)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("generate", generate)
graph.add_node("direct_answer", direct_answer)

# 3) NORMAL EDGE: always go START → decide_to_retrieve.
graph.add_edge(START, "decide_to_retrieve")

# 4) CONDITIONAL EDGE: after decide_to_retrieve, choose_branch() returns
#    a key from this dict, and we jump to that node.
graph.add_conditional_edges(
    "decide_to_retrieve",  # from this node
    choose_branch,  # function that picks the branch
    {
        "retrieve": "retrieve",  # if it returns "retrieve", go here
        "direct_answer": "direct_answer",  # if it returns "direct_answer", go here
    },
)

# 5) NORMAL EDGE: retrieve always goes to the grader (never straight to generate).
graph.add_edge("retrieve", "grade_documents")

# 6) CONDITIONAL EDGE: grader decides generate vs rewrite-and-retry.
graph.add_conditional_edges(
    "grade_documents",
    choose_after_grade,
    {
        "generate": "generate",
        "rewrite": "rewrite_question",
    },
)

# 7) NORMAL EDGES: rewrite always retrieves again; answers always stop at END.
graph.add_edge("rewrite_question", "retrieve")
graph.add_edge("generate", END)
graph.add_edge("direct_answer", END)

# 6) compile() turns the drawings above into something you can .invoke().
app = graph.compile()


if __name__ == "__main__":
    load_dotenv()
    print("Chat started (type quit to exit).\n")

    # We hold the message list in this variable and feed it back each turn.
    messages = []
    first = " ".join(sys.argv[1:])
    while True:
        question = first or input("You: ").strip()
        first = ""
        if not question or question.lower() in {"quit", "exit"}:
            break

        result = app.invoke(
            {
                "question": question,
                "messages": [*messages, HumanMessage(content=question)],
            }
        )
        messages = result["messages"]

        print(f"\nAnswer ({len(messages)} messages stored):")
        print(result["answer"])
        print()
        for i, (chunk, score) in enumerate(result.get("results") or [], start=1):
            print(f"--- chunk {i} | score: {score:.4f} ---")
            print(chunk)
            print()
