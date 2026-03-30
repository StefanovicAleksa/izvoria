# Canvas: visual_cue_prompts_v3
# Path: app/features/visual_cue_detection/data/prompts.py

VISUAL_CUE_SYSTEM_PROMPT = """
You are an expert NLP gatekeeper for an Enterprise Legal and Medical AI.
Your job is to read a text chunk and its surrounding context, and determine if the speaker is referencing visual information.

You must set `needs_visual` to TRUE if:
- The speaker uses explicit deictic expressions (e.g., "this chart", "look here", "that bone", "Exhibit A").
- IMPLICIT VISUALS: The speaker narrates spatial movement, financial chart action (e.g., "finding support at 20", "dipping through 19s"), geometric shapes, or physical actions that require seeing the screen to fully understand.

You must set `needs_visual` to FALSE if:
- It is general conversation, a talking head, or an abstract concept without spatial/visual references.
- The speaker is telling a story without pointing to physical evidence.
"""

# Strict JSON schema enforced by llama-cpp-python at the inference level
CUE_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_visual": {
            "type": "boolean",
            "description": "True if visual context is required to understand the speaker's references."
        }
    },
    "required": ["needs_visual"]
}