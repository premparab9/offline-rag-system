from langchain_ollama import OllamaLLM
from config import LLM_MODEL

def get_llm(model_name: str = None):
    """
    Return a LangChain LLM wrapper for Ollama.

    Args:
        model_name: Ollama model to use (e.g. "gemma2:2b").
                    Falls back to config default if not provided.
    """
    model = model_name or LLM_MODEL

    return OllamaLLM(
        model=model,
        temperature=0.1,      # Low = more deterministic answers
        num_ctx=4096,         # Context window size
    )


def generate_response(prompt: str, model_name: str = None) -> str:
    """
    Send a prompt to the local LLM and return the response.

    Args:
        prompt: The full prompt string (context + question)
        model_name: Optional override for which model to use
    """
    llm = get_llm(model_name)

    # .invoke() sends the prompt and waits for full response
    response = llm.invoke(prompt)
    return response