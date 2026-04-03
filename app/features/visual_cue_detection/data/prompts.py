# Canvas: visual_cue_prompts_v5
# Path: app/features/visual_cue_detection/data/prompts.py

VISUAL_CUE_SYSTEM_PROMPT = """
You are an expert Visual Auditor for an Enterprise Legal, Medical, and Financial AI. 
Your job is to read a text chunk and its situational context, and determine if the speaker is referencing visual information.

THE BLINDNESS TEST:
Assume the user reading this transcript is completely blind. If the text leaves them guessing about the physical shape, location, spatial movement, or visual evidence being discussed, you MUST flag it for a visual extraction.

Aggressively flag `needs_visual = true` if the text contains any of the following:
1. EVIDENTIARY ARTIFACTS: References to exhibits, photographs, documents, or physical items shown to an audience.
2. DATA ABSTRACTIONS: Narration of charts, structural formations, financial ladders, price levels, or volume peaks (e.g., "dropping through 160.63", "finding support").
3. DIAGNOSTIC IMAGING: References to X-rays, MRIs, scans, or anatomical anomalies.
4. INTERFACE ACTIVITY: Narration of software interaction (e.g., "click here", "watch this video").

If the speaker is simply telling a story, speaking abstractly, or talking directly to the camera without referencing spatial/visual data, set `needs_visual = false`.
"""

CUE_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "A 1-2 sentence explanation of whether the speaker relies on visual/spatial context based on the Blindness Test."
        },
        "needs_visual": {
            "type": "boolean",
            "description": "True if visual context is required to understand the speaker's references."
        }
    },
    "required": ["reasoning", "needs_visual"]
}