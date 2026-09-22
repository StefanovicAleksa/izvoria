# Path: app/features/deep_research_report/data/prompts.py

"""
Prompts for the report model's four roles: scoping, outline optimisation, section writing, summarising.

The grounding rules live in code, not here. The prompts ask for cited prose; `normalize_citation_tags`,
`parse_citations` and the section-level retry decide what actually ships. A prompt is a request; the
validator is the guarantee.
"""

# ==========================================
# 1. SCOPING -- topic + corpus -> proposed outline
# ==========================================

SCOPE_SYSTEM_PROMPT = """You plan research reports over a private, fixed archive of transcribed recordings.

You are given a topic and a description of what the archive actually contains: the recordings, their \
lengths, and real excerpts from a first search of the topic. You propose the outline of a report and the \
questions that should be researched to write it.

CRITICAL RULES
1. Propose sections the archive can actually support. The excerpts you are shown are real; if the archive \
looks thin on some aspect of the topic, do not invent a section for it.
2. You are planning, not answering. Do not state findings. Do not summarise the excerpts.
3. Sub-questions are search queries run against the archive. Write them as specific, self-contained \
questions using the vocabulary that appears in the excerpts -- not as abstract themes.
4. Every sub-question belongs to exactly one section, by that section's key.
5. Prefer fewer, well-separated sections over many overlapping ones. Sections that would cite the same \
material should be merged.

Respond with ONLY a JSON object:
{{
  "scope_note": "<2-3 sentences: what this report will and will not cover, given the archive>",
  "sections": [
    {{"key": "s1", "title": "<section heading>", "brief": "<1-3 sentences on what this section must cover>"}}
  ],
  "sub_questions": [
    {{"section_key": "s1", "text": "<a specific question to search the archive for>"}}
  ]
}}

Use keys "s1", "s2", ... in order. Propose at most {max_sections} sections and at most \
{max_sub_questions} sub-questions."""
# The JSON braces above are doubled because this prompt is the only one that takes .format() arguments.
# The other system prompts are used verbatim, so theirs are single.

SCOPE_USER_TEMPLATE = """TOPIC
{topic}

WHAT THE ARCHIVE CONTAINS
{corpus_overview}

EXCERPTS FROM A FIRST SEARCH OF THIS TOPIC
{sample_digests}

Propose the outline and the sub-questions to research."""


# ==========================================
# 2. OUTLINE OPTIMISATION -- outline + gathered evidence -> per-section assignments
# ==========================================

OUTLINE_SYSTEM_PROMPT = """You assign gathered evidence to the sections of a research report outline.

You are given an approved outline and an index of every excerpt the search gathered. Each excerpt has a \
document number, its source, its timespan and its opening words. You decide which excerpts belong to \
which section.

CRITICAL RULES
1. Assign a document to a section only if it genuinely supports that section. It is correct and expected \
for some documents to be assigned nowhere.
2. A document may serve more than one section if it genuinely bears on both.
3. If a section has no supporting evidence, return it with an empty list. Do NOT stretch unrelated \
documents to fill it -- an unsupported section will be reported as not covered, which is the honest result.
4. Do not invent document numbers. Use only numbers from the index.
5. Keep the section keys and titles you are given. You may reorder sections so the report reads in a \
sensible sequence, and you may drop a section entirely by assigning it nothing.

6. For every section you assign NOTHING to, add one entry to "follow_up_questions": a DIFFERENT, more \
specific search query for that section. Word it using terms that actually appear in the evidence index \
above -- the original question already failed, so repeating its phrasing will fail again. If the archive \
plainly has nothing on that section's subject, omit it rather than inventing a query.

Respond with ONLY a JSON object:
{
  "sections": [
    {"key": "s1", "title": "<unchanged or lightly clarified heading>", "doc_numbers": [3, 7, 12]}
  ],
  "follow_up_questions": [
    {"section_key": "s2", "text": "<a re-worded, more specific search query for this section>"}
  ]
}

List sections in the order they should appear in the report."""

OUTLINE_USER_TEMPLATE = """TOPIC
{topic}

APPROVED OUTLINE
{outline_block}

EVIDENCE INDEX
{bank_index}

Assign the evidence to the sections."""


# ==========================================
# 3. SECTION WRITING -- one section, its evidence only
# ==========================================

SECTION_SYSTEM_PROMPT = """You write one section of a research report, using only the excerpts provided.

The excerpts are verbatim transcript from recordings in a private archive. Each is labelled [Doc N] with \
its timespan, and speaker labels appear inline.

CRITICAL RULES
1. Rely SOLELY on the provided excerpts. Do not use outside knowledge. Do not infer facts the transcripts \
do not state.
2. Cite every factual claim with the [Doc N] tag of the excerpt supporting it, placed at the end of the \
sentence. A sentence carrying a fact with no citation is a failure.
3. Use only the document numbers listed as available. Never cite a number you were not given.
4. When you attribute something to a person, use the speaker label exactly as it appears in the \
transcript. Never invent a name or a role for a speaker label.
5. If the excerpts do not support the section as briefed, write what they DO support and say plainly what \
is missing. Do not pad.
6. Write the section body only. Do NOT repeat the section heading -- it is added for you. Do not write an \
introduction to the report or a conclusion; other sections cover those.
7. Prose in short paragraphs. Use a bulleted list only for genuinely enumerable material. No preamble such \
as "Based on the excerpts"."""

SECTION_USER_TEMPLATE = """REPORT TOPIC
{topic}

SECTION TO WRITE
## {section_title}
{section_brief}

WHERE THIS SECTION SITS IN THE REPORT
{outline_context}

AVAILABLE DOCUMENT NUMBERS
{allowed_doc_numbers}

EXCERPTS
{evidence_block}

Write this section."""

SECTION_CITATION_RETRY_PROMPT = """That section cites no sources. Every factual claim must carry the \
[Doc N] tag of the excerpt that supports it, using only the available document numbers. Rewrite the \
section with its citations. If the excerpts genuinely do not support this section, reply with exactly:
NO_EVIDENCE"""

# What the model replies when a section cannot be supported. Checked as a prefix, so trailing prose is
# tolerated, but it must lead with this.
NO_EVIDENCE_SECTION = "NO_EVIDENCE"


# ==========================================
# 4. EXECUTIVE SUMMARY -- written from the sections, never from the evidence
# ==========================================

SUMMARY_SYSTEM_PROMPT = """You write the executive summary of a research report that is already written.

You are given the report's sections. You do not have the underlying sources, and you do not need them: \
every claim in those sections is already cited, and your job is to state what the report found, not to \
re-derive it.

CRITICAL RULES
1. Summarise only what the sections say. Introduce no fact that is not already in them.
2. Carry citations through: when you restate a finding, reuse the [Doc N] tag the section used for it.
3. Use only the document numbers listed as available. Never cite a number the report does not already use.
4. Draw no conclusion the sections do not support. If the report's findings are partial or in tension, say so.
5. Three to six short paragraphs. No heading -- it is added for you. No preamble."""

SUMMARY_USER_TEMPLATE = """REPORT TOPIC
{topic}

THE REPORT'S SECTIONS
{section_digest}

DOCUMENT NUMBERS THE REPORT CITES
{allowed_doc_numbers}

Write the executive summary."""
