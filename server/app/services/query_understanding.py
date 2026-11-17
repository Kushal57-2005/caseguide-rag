import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_LLM_API_KEY"),
    base_url=os.getenv("NVIDIA_LLM_BASE_URL")
)

LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")


def extract_json_from_text(text: str) -> dict:
    """Safely extracts and parses JSON from text, handling markdown fences and common issues."""
    if not text:
        return {}
    
    # Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Try extracting inside ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    # Try finding outermost { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1].strip())
        except Exception:
            pass

    return {}


def understand_query(original_query: str) -> dict:
    """
    Performs LLM-based query understanding before document retrieval.
    Generates:
      - refined_query: query optimized for legal document retrieval
      - plain_language_query: simplified plain-English summary of what the user is asking
      - title: concise title of the legal issue
      - category: relevant legal domain
      - description: brief explanation of user's situation
    """
    system_prompt = """You are an expert legal query analyzer.
Your task is to analyze the user's question and produce a structured JSON object.

Return valid JSON with exactly these keys:
{
  "refined_query": "An improved, precise, search-oriented version of the question with key statutory terms, legal concepts, and keywords optimized for dense legal vector retrieval",
  "plain_language_query": "A clear, simplified version of what the user is asking in everyday plain language",
  "title": "A short, concise title summarizing the legal problem (under 8 words)",
  "category": "The specific legal domain or category (e.g. Motor Vehicles & Traffic Law, Criminal Negligence, Accident Compensation)",
  "description": "A brief 1-2 sentence explanation of the user's problem and situation"
}

Important:
- Preserve the user's actual intent.
- Do not invent facts that the user did not state.
- Output ONLY valid JSON.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User question:\n{original_query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )

        message = response.choices[0].message
        content = message.content or getattr(message, "reasoning_content", None) or ""
        parsed = extract_json_from_text(content)

        return {
            "refined_query": str(parsed.get("refined_query") or original_query).strip(),
            "plain_language_query": str(parsed.get("plain_language_query") or original_query).strip(),
            "title": str(parsed.get("title") or "Legal Question Analysis").strip(),
            "category": str(parsed.get("category") or "General Legal").strip(),
            "description": str(parsed.get("description") or original_query).strip()
        }

    except Exception as e:
        # Graceful fallback without breaking pipeline
        return {
            "refined_query": original_query.strip(),
            "plain_language_query": original_query.strip(),
            "title": "Legal Issue Inquiry",
            "category": "General Legal",
            "description": original_query.strip()
        }
