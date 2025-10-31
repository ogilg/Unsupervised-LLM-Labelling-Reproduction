"""
Wrapper functions for Hyperbolic API Llama 3.1 405B models.
"""
import os
import time
from openai import OpenAI, APIError, InternalServerError
from dotenv import load_dotenv


class HyperbolicClient:
    """Wrapper client for Hyperbolic API models."""

    def __init__(self, api_key: str = None, model_type: str = "base"):
        """
        Initialize Hyperbolic client.

        Args:
            api_key: API key for Hyperbolic. If None, loads from environment.
            model_type: Type of model to use ("base" or "instruct"). Defaults to "base".
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("HYPERBOLIC_API_KEY")

        if model_type not in ["base", "instruct"]:
            raise ValueError(f"model_type must be 'base' or 'instruct', got {model_type}")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.hyperbolic.xyz/v1"
        )
        self.model_type = model_type

        # Assign model name based on type
        if model_type == "base":
            self.model_name = "meta-llama/Meta-Llama-3.1-405B"
        else:  # instruct
            self.model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        logprobs: int = 5,
        temperature: float = 0.0,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0
    ):
        """
        Generate completion using configured model type with retry logic.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            logprobs: Number of top logprobs to return per token (0 to disable)
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts for API errors
            initial_retry_delay: Initial delay in seconds before first retry (doubles each retry)

        Returns:
            Response object with completion and optionally logprobs
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if self.model_type == "instruct":
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        logprobs=logprobs > 0,
                        top_logprobs=logprobs if logprobs > 0 else None,
                        temperature=temperature
                    )
                else:  # base
                    return self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        logprobs=logprobs if logprobs > 0 else None,
                        temperature=temperature
                    )
            except (InternalServerError, APIError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = initial_retry_delay * (2 ** attempt)
                    print(f"\nAPI error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"\nMax retries ({max_retries}) reached. Giving up.")
        
        raise last_exception

    def get_text(self, response) -> str:
        """Extract generated text from response based on model type."""
        if self.model_type == "instruct":
            return response.choices[0].message.content
        else:  # base
            return response.choices[0].text

    def get_logprobs(self, response):
        """Extract logprobs from response based on model type."""
        if self.model_type == "instruct":
            if response.choices[0].logprobs is None:
                return None
            return response.choices[0].logprobs.content
        else:  # base
            return response.choices[0].logprobs
