"""
Teacher LLM Client (OpenAI-compatible).

Provides a generic client for any OpenAI-compatible API (loopback URLs).
"""
import logging
from typing import Optional, List, Dict, Any
import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


class TeacherClient:
    """
    Generic client for OpenAI-compatible APIs.

    Uses loopback URLs (localhost/LAN) to access Claude/GPT via proxies.

    Example:
        >>> client = TeacherClient(
        ...     base_url="http://localhost:8001/v1",
        ...     api_key="sk-xxx",
        ...     model="claude-3-5-sonnet-20241022"
        ... )
        >>> response = client.generate("What is 2+2?")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize Teacher Client.

        Args:
            base_url: OpenAI-compatible API base URL (e.g., http://localhost:8001/v1)
            api_key: API key for the service
            model: Model name (e.g., "claude-3-5-sonnet-20241022")
            timeout: Request timeout in seconds
            max_retries: Max number of retries on failure
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize OpenAI client with custom base_url
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(f"Teacher client initialized: {model} @ {base_url}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters passed to API

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generate JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Parsed JSON dict
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that responds in JSON format."

        # Add JSON instruction to prompt
        full_prompt = f"{prompt}\n\nPlease respond in valid JSON format."

        response_text = self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse JSON
        import json
        try:
            # Try to extract JSON from code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response text: {response_text}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ~= 4 chars
        return len(text) // 4

    def __repr__(self) -> str:
        return f"TeacherClient(model={self.model}, base_url={self.base_url})"


class MultiSourceTeacher:
    """
    Multi-source teacher that aggregates multiple LLM teachers.

    Example:
        >>> teacher = MultiSourceTeacher()
        >>> teacher.add_client("claude", claude_client)
        >>> teacher.add_client("gpt", gpt_client)
        >>> responses = teacher.generate_diverse("What is 2+2?", num_per_source=2)
    """

    def __init__(self):
        """Initialize multi-source teacher."""
        self.clients: Dict[str, TeacherClient] = {}
        logger.info("Multi-source teacher initialized")

    def add_client(self, name: str, client: TeacherClient):
        """
        Add a teacher client.

        Args:
            name: Client name (e.g., "claude", "gpt")
            client: TeacherClient instance
        """
        self.clients[name] = client
        logger.info(f"Added teacher client: {name}")

    def generate_diverse(
        self,
        prompt: str,
        num_per_source: int = 1,
        temperature: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse responses from all sources.

        Args:
            prompt: User prompt
            num_per_source: Number of generations per source
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            List of dicts with keys: "source", "content", "tokens"
        """
        results = []

        for source_name, client in self.clients.items():
            for i in range(num_per_source):
                try:
                    content = client.generate(
                        prompt=prompt,
                        temperature=temperature,
                        **kwargs
                    )

                    results.append({
                        "source": source_name,
                        "content": content,
                        "tokens": client.count_tokens(content),
                        "index": i,
                    })

                except Exception as e:
                    logger.error(f"Failed to generate from {source_name}: {e}")

        return results

    def __repr__(self) -> str:
        sources = ", ".join(self.clients.keys())
        return f"MultiSourceTeacher(sources=[{sources}])"
