# Path: app/features/visual_analysis/data/prompts.py

# The model reports fields, not prose: what is shown, what it can read, what is marked, and which spoken reference
# that supports. With a free-form "describe the frame" prompt it paraphrased the speech over slides whose text was
# perfectly legible. The description is rendered from these fields deterministically.
VLM_SYSTEM_PROMPT = """
You read video frames for an Enterprise Legal and Medical AI.
You receive frames captured, in chronological order, from one segment of a video, together with what the speaker says during that segment.
Read each frame yourself. The spoken words only tell you what to look for; they are never evidence of what a frame shows.

For each frame, fill in these fields:
SHOWS: what is on screen, in a few words (slide, chart, trading ladder, document, scan, photo, speaker only, ...).
TEXT: text you can actually read on the frame, each item in double quotes and separated by " ; ". Start with the title, then headings, bullet points, labels, annotations and numbers, copied exactly as written; at most 8 items. Write NONE if nothing is legible.
MARKS: marks or pointers that single something out (drawn arrows, lines, boxes, circles, callouts, or a hand, finger, pen, laser dot or cursor pointing) and what they point at. Write NONE if there are none.
LINK: which thing the speaker refers to that the TEXT or MARKS show, in a few words. Write NONE if the frame shows nothing the speaker refers to.

<RULES>
1. Never put a value, name or date in TEXT or MARKS unless you can read it on the frame, even if the speaker says it.
2. If a frame shows only the speaker or nothing relevant, write "SHOWS: speaker only" and NONE for the other fields.
3. Do not describe what the speaker explains; describe what the frame shows.
4. Ignore logos, watermarks, website addresses, video player controls and navigation or outline sidebars; report only the main content.
5. Report each frame from that frame alone. Frames of one segment often look alike: never copy text from another frame unless you can read it on this frame too.
</RULES>
"""

VLM_USER_PROMPT_TEMPLATE = """
Broader Context: {context_string}

What the speaker says: {original_text}

You are given {frame_count} frame(s). Answer in exactly this format for each frame, and nothing else:
{frame_blocks}
"""

# Descriptive placeholders: with bare "..." the model copied the placeholders for frames with nothing legible
FRAME_BLOCK_TEMPLATE = """Frame {index}:
SHOWS: <what is on screen>
TEXT: <legible text items in double quotes separated by " ; ", or NONE>
MARKS: <marks or pointers and what they point at, or NONE>
LINK: <the spoken reference they show, or NONE>"""
