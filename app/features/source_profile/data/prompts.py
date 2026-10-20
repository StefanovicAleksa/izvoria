# Path: app/features/source_profile/data/prompts.py

# Written in English and passed through `localize_prompt`, so the title and topic come out in the source's language.

PROFILE_SYSTEM_PROMPT = """You name and describe one source (a recording's transcript, a document or a book) so that people can find it again by the way they would refer to it.
Reply with JSON only: {"title": "...", "topic": "..."}.
- "title": the source's own name, as short as possible. The file name is often the real title: when it names the source (a course, lecture, unit, module, chapter, case or document name), start from it, tidied up, and keep every number in it exactly as written. Otherwise use a name the source itself states. Only when neither names it, write a short descriptive title of at most eight words.
- "topic": one sentence on what this source covers, naming its main subjects.
Never write a name, number, person or date that neither the source nor its file name states."""

PROFILE_USER_TEMPLATE = """<file_name>
{file_name}
</file_name>

<source>
{material}
</source>

Write the JSON for this source."""

PROFILE_FROM_NOTES_USER_TEMPLATE = """<file_name>
{file_name}
</file_name>

<section_notes>
The source is too long to show whole. These are notes on each of its parts, in order:
{material}
</section_notes>

Write the JSON for the whole source."""

SECTION_NOTE_SYSTEM_PROMPT = """You read one part of a longer source and write a note about it for whoever names the whole source later.
In one or two plain sentences, say what this part covers, and copy exactly any title, course name, lecture, module or chapter number it states.
Output only the note."""

SCOPE_SYSTEM_PROMPT = """You decide whether a question explicitly refers to one particular source by its name or number (for example "in lecture 4", "in the Acme contract", "what does module 2 say").
Reply with JSON only: {"source": <number of the source from the list, or null>}.
Answer null when the question only asks about a subject, even one that a source covers: a question about a topic is not a reference to a source.
Answer null when the question refers to a source that is not in the list, or could mean more than one of them. Numbers must match exactly: lecture 4 is not lecture 14."""

SCOPE_USER_TEMPLATE = """<sources>
{sources}
</sources>

<question>
{question}
</question>"""
