import json
import re
from typing import Dict, Any

from app.features.asr_error_correction.exceptions.pipeline import GlossaryJSONParseError


def extract_and_parse_json(raw_llm_output: str) -> Dict[str, Any]:
    """
    Aggressively strips markdown, conversational padding, and hallucinated wrappers
    to extract the raw JSON payload from Phase 1 (Llama 3.2).
    """
    text = raw_llm_output.strip()

    # 1. Try to find JSON markdown blocks
    json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        text = json_block_match.group(1).strip()
    else:
        # 2. Fallback: Find the first '{' and the last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            text = text[start_idx:end_idx + 1]

    try:
        parsed_data = json.loads(text)
        if not isinstance(parsed_data, dict):
            raise GlossaryJSONParseError("Parsed LLM output is not a JSON object.")
        return parsed_data
    except json.JSONDecodeError as e:
        raise GlossaryJSONParseError(
            f"Failed to decode extracted JSON: {str(e)}\nRaw LLM Output: {raw_llm_output}") from e