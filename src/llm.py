"""
LLM Integration module using Google Gemini.
Supports both Google's modern `google-genai` SDK and `google-generativeai`.
Synthesizes accurate answers grounded strictly in retrieved PDF context,
preventing hallucinations and enforcing page citations.
"""

from dataclasses import dataclass, field
import logging
import os
from typing import List, Optional
import warnings
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Suppress deprecation warnings if fallback is used
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# Check which Gemini client library is available
USE_MODERN_GENAI = False
try:
    from google import genai
    from google.genai import types
    USE_MODERN_GENAI = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
    except ImportError:
        pass

SYSTEM_PROMPT = """You are an expert AI document research assistant. Your task is to answer the user's question using ONLY the provided document context below.

CRITICAL INSTRUCTIONS:
1. Answer accurately and strictly based on the context provided.
2. If the answer cannot be found in the provided context, or if the context is insufficient, state clearly: "I could not find information about that in the uploaded documents."
3. Do NOT make assumptions, extrapolate without stating so, or invent/fabricate facts, data, statistics, or page citations.
4. Prefer clear, well-structured, and concise explanations (using bullet points where helpful).
5. Explicitly cite the source document name and page number for each key point mentioned (e.g. [Document Name, Page X]).
6. Clearly distinguish between confirmed facts from the text and any ambiguities.
"""


@dataclass
class SourceCitation:
    """Document citation for an answer."""
    document_name: str
    page_number: int
    chunk_id: str
    similarity_score: float
    excerpt: str


@dataclass
class LLMResponse:
    """Standardized response from the LLM synthesis pipeline."""
    answer: str
    citations: List[SourceCitation] = field(default_factory=list)
    model_name: str = ""
    prompt_tokens: Optional[int] = None
    candidates_tokens: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class GeminiRAGGenerator:
    """
    Handles prompt construction and inference with Google Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ):
        """
        Initialize Gemini generator.

        Args:
            api_key: Google Gemini API key. Defaults to GEMINI_API_KEY env variable.
            model_name: Gemini model ID (e.g. 'gemini-2.5-flash', 'gemini-2.5-pro').
            temperature: Sampling temperature (low for factual grounding).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        # Automatically map retired legacy model names to current active equivalents
        if "1.5" in model_name or "2.0" in model_name:
            model_name = "gemini-2.5-flash"
        self.model_name = model_name
        self.temperature = temperature
        self._modern_client = None
        self._configured = False

        if self.api_key and not self.api_key.startswith("your_"):
            self._configure_api()

    def _configure_api(self):
        """Configure Gemini client."""
        try:
            if USE_MODERN_GENAI:
                self._modern_client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": "v1beta"},
                )
                self._configured = True
            else:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._configured = True
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            self._configured = False

    def update_api_key(self, api_key: str):
        """Dynamically update API key at runtime."""
        self.api_key = api_key.strip()
        self._configure_api()

    def _call_model(
        self,
        model_name: str,
        user_prompt: str,
        citations: List[SourceCitation],
    ) -> LLMResponse:
        """Call Gemini API using either modern or legacy SDK."""
        if USE_MODERN_GENAI and self._modern_client:
            config_obj = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
            )
            response = self._modern_client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config_obj,
            )
            answer_text = response.text if response and response.text else "No response generated."
        else:
            import google.generativeai as legacy_genai
            model = legacy_genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                generation_config=legacy_genai.GenerationConfig(
                    temperature=self.temperature,
                ),
            )
            response = model.generate_content(user_prompt)
            answer_text = response.text if response and response.text else "No response generated."

        return LLMResponse(
            answer=answer_text,
            citations=citations,
            model_name=model_name,
            success=True,
        )

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> LLMResponse:
        """
        Synthesize an answer to query based on retrieved chunks.

        Args:
            query: User's question.
            retrieved_chunks: List of top_k RetrievedChunk objects.

        Returns:
            LLMResponse with answer, citations, and error handling.
        """
        if not self.api_key or self.api_key.startswith("your_"):
            return LLMResponse(
                answer=(
                    "⚠️ **Google Gemini API Key is missing or invalid.**\n\n"
                    "Please set your `GEMINI_API_KEY` in the `.env` file or enter it "
                    "in the sidebar settings."
                ),
                citations=[],
                model_name=self.model_name,
                success=False,
                error_message="Missing GEMINI_API_KEY",
            )

        if not self._configured:
            self._configure_api()
            if not self._configured:
                return LLMResponse(
                    answer="⚠️ Failed to configure Google Gemini API client. Please verify your API key.",
                    citations=[],
                    model_name=self.model_name,
                    success=False,
                    error_message="Gemini API configuration failed.",
                )

        # Build citations list from chunks
        citations: List[SourceCitation] = []
        unique_seen = set()
        for chunk in retrieved_chunks:
            key = (chunk.document_name, chunk.page_number)
            if key not in unique_seen:
                unique_seen.add(key)
                citations.append(
                    SourceCitation(
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        chunk_id=chunk.chunk_id,
                        similarity_score=chunk.similarity_score,
                        excerpt=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    )
                )

        # Handle empty context
        if not retrieved_chunks:
            return LLMResponse(
                answer=(
                    "I could not find any relevant information in the uploaded documents to answer your question. "
                    "Please make sure relevant documents are uploaded or try rephrasing your question."
                ),
                citations=[],
                model_name=self.model_name,
                success=True,
            )

        # Format context blocks with clear page and source boundaries
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            block = (
                f"--- CONTEXT BLOCK {idx} ---\n"
                f"Document: {chunk.document_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Content:\n{chunk.text}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        user_prompt = f"""Context from uploaded documents:
{context_str}

User Question:
{query}

Answer the question strictly based on the context above. Cite the source document name and page number for each fact:"""

        try:
            return self._call_model(self.model_name, user_prompt, citations)

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Gemini API generation error with '{self.model_name}': {error_msg}")

            # If model name failed due to 404/NOT_FOUND, automatically retry with 'gemini-2.5-flash'
            if ("404" in error_msg or "NOT_FOUND" in error_msg) and self.model_name != "gemini-2.5-flash":
                try:
                    logger.info("Attempting automatic fallback to 'gemini-2.5-flash'...")
                    return self._call_model("gemini-2.5-flash", user_prompt, citations)
                except Exception as fallback_err:
                    error_msg = str(fallback_err)

            if "API_KEY_INVALID" in error_msg or "invalid api key" in error_msg.lower():
                user_msg = "⚠️ The provided Gemini API Key is invalid. Please check your key in `.env` or sidebar."
            elif "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                user_msg = "⚠️ Gemini API rate limit or quota exceeded. Please wait a moment or check your Google AI Studio quota."
            else:
                user_msg = f"⚠️ Gemini API generation failed: {error_msg}"

            return LLMResponse(
                answer=user_msg,
                citations=citations,
                model_name=self.model_name,
                success=False,
                error_message=error_msg,
            )
