import config
from functools import lru_cache
from langchain_ollama import ChatOllama

# @lru_cache(maxsize=2)
# def get_llm(**kwargs) -> ChatOllama:

#     output_format = kwargs.pop("format", None)

#     llm_kwargs = {
#         "model": config.MODEL_URI,
#         "temperature": 0,
#         "base_url": config.OLLAMA_BASE_URL,
#     }
#     if output_format is not None:
#         llm_kwargs["format"] = output_format

#     llm_kwargs.update(kwargs)

#     return ChatOllama(**llm_kwargs)


@lru_cache(maxsize=4)
def _build_llm(model_uri: str, temperature: float, json_mode: bool) -> ChatOllama:
    """Internal builder, cached by (model, temperature, json_mode)."""
    return ChatOllama(
        model=model_uri,
        temperature=temperature,
        base_url=config.OLLAMA_BASE_URL,
        format="json" if json_mode else None,
    )


def get_llm(format: str | None = None, temperature: float = 0.0) -> ChatOllama:
    """
    Return a shared, cached ChatOllama client.
    Two distinct clients are cached:
      - one for plain text generation
      - one for structured JSON output
    No new HTTP client is created on subsequent calls.
    """
    return _build_llm(
        config.MODEL_URI,
        temperature,
        json_mode=(format == "json"),
    )
