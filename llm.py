
from langchain_ollama import OllamaLLM
from config import LLM_MODEL
from logger import get_logger

log = get_logger(__name__)

AVAILABLE_MODELS = {
    "gemma2:2b": {
        "label":       "Gemma 2 · 2B  (SLM — Fast)",
        "type":        "slm",
        "description": "Small and fast. Good for straightforward Q&A.",
    },
}

_SIMPLE_STARTS = frozenset([
    "what is", "what are", "who is", "who are", "when did", "when was",
    "where is", "where are", "define", "list", "name", "how many", "which",
])

_COMPLEX_KEYWORDS = frozenset([
    "explain", "summarize", "summarise", "compare", "analyse", "analyze",
    "describe", "elaborate", "discuss", "why did", "why does", "how does",
    "what would happen", "give me a detailed", "walk me through",
])


def route_query(question, selected_model):
    available = list(AVAILABLE_MODELS.keys())

    if len(available) == 1:
        return available[0], f"Only one model available: {available[0]}"

    q = question.lower().strip()

    for phrase in _SIMPLE_STARTS:
        if q.startswith(phrase):
            slms = [k for k, v in AVAILABLE_MODELS.items() if v["type"] == "slm"]
            if slms:
                return slms[0], f"Simple question — using fast SLM"

    for keyword in _COMPLEX_KEYWORDS:
        if keyword in q:
            llms = [k for k, v in AVAILABLE_MODELS.items() if v["type"] == "llm"]
            if llms:
                return llms[0], f"Complex question — using larger LLM"

    return selected_model, "Using model selected in sidebar"


def generate_response(prompt, model_name=None):
    log.info("Generating with model: %s", model_name or LLM_MODEL)
    llm = OllamaLLM(model=model_name or LLM_MODEL, temperature=0.1, num_ctx=4096)
    return llm.invoke(prompt)
