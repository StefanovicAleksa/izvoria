# Path: app/core/language/domain/entities.py

"""
A language pack: every rule in Onyx that depends on the language being spoken, in one validated object.

An appliance ingests and answers in every language that has a pack in `data/languages/<code>.toml`; one of
them is active at a time (`settings.APPLIANCE_LANGUAGE`). Nothing here translates anything: the transcript, the index and the answers all
stay in the language that was spoken, so a citation always points at the words a person actually said.
"""

import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The refusal sentence as the English prompts spell it. Localising a prompt swaps it for the pack's own.
ENGLISH_REFUSAL = "I cannot answer this based on the provided context."


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _compiles(pattern: str) -> str:
    """A regex from a pack file must compile; re.error is not a ValueError, so pydantic would not report it."""
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regular expression {pattern!r}: {e}") from e
    return pattern


class TranscriptionRules(_Frozen):
    whisper_language: str
    # Whisper labels that count as this language when the guard checks a file. Serbian audio is routinely
    # detected as Croatian or Bosnian, so a strict equality check would reject correct files.
    accepted_detections: List[str]
    # A detection below this probability is too unsure to reject a file on.
    min_detection_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    # Script every transcript and query is normalised to. Only "latin" (Serbian Cyrillic -> Latin) exists.
    script: Optional[Literal["latin"]] = None
    # Phrases Whisper invents in this language over silence or music, removed as hallucinations.
    hallucination_patterns: List[str] = Field(default_factory=list)

    @field_validator("hallucination_patterns")
    @classmethod
    def _patterns_compile(cls, patterns: List[str]) -> List[str]:
        for pattern in patterns:
            _compiles(pattern)
        return patterns


class PunctuationRules(_Frozen):
    # "model": the fullstop token classifier re-punctuates Whisper's words (en/de/fr/it only).
    # "whisper": keep the punctuation Whisper wrote, for languages the classifier was never trained on.
    source: Literal["model", "whisper"]


class SearchRules(_Frozen):
    # Postgres text search configuration. Used in DDL, so restricted to identifiers.
    fts_config: str = Field(pattern=r"^[a-z_]+$")


class TextRules(_Frozen):
    # Resolved by the repository: an inline list, or the words of an NLTK stopword corpus.
    stopwords: Tuple[str, ...] = ()
    # Words allowed between the numbers of one citation group: "[Doc 1 and 2]".
    citation_joiners: List[str] = Field(min_length=1)


class PhoneticRules(_Frozen):
    # Lowercase letters of the alphabet as a regex character-class body, e.g. "a-z" or "a-zčćđšž".
    alphabet: str
    # Spoken letter names, so a spelled-out acronym can be compared with the acronym.
    letter_names: Dict[str, str]
    # Applied before anything else: letters folded to the plain letter they are most often mis-heard as.
    fold: Dict[str, str] = Field(default_factory=dict)
    # Ordered spelling-to-sound rewrites; later rules see the output of earlier ones.
    rewrites: List[Tuple[str, str]]
    # Letters dropped from a key after its first letter.
    silent_letters: str
    # A key starting with one of these starts with "a" instead, so leading vowels compare equal.
    head_vowels: str
    # Plural ending the matcher may add or drop to keep the speaker's number ("burns" -> "Bunds"). Empty for an
    # inflected language, whose case endings a glossary term can't reproduce: the term is then used as written.
    plural_suffix: str = ""
    # A heard word with one of these endings is never replaced by a term without it ("burned" -> "Bund").
    verb_suffixes: List[str] = Field(default_factory=list)

    @field_validator("alphabet")
    @classmethod
    def _alphabet_compiles(cls, alphabet: str) -> str:
        _compiles(f"[{alphabet}]")
        return alphabet

    @field_validator("rewrites")
    @classmethod
    def _rewrites_compile(cls, rewrites: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        for pattern, _ in rewrites:
            _compiles(pattern)
        return rewrites


class GlossaryRules(_Frozen):
    # Wikidata label languages the glossary miner collects terms in.
    wikidata_languages: List[str] = Field(min_length=1)


class AnswerRules(_Frozen):
    # The grounded refusal, in this language.
    refusal: str
    # Appended to prompts whose output is stored or shown; empty for English, whose prompts need nothing.
    output_instruction: str = ""
    # Appended to prompts that edit a transcript in place.
    transcript_instruction: str = ""


class ReportStrings(_Frozen):
    """The report's fixed text. Templates take named fields; a template may leave any of them out."""
    executive_summary: str
    not_covered_heading: str
    not_covered_item: str
    references_heading: str
    references_table_header: str
    # The same header for a report citing a document, whose location column holds pages, not times
    references_table_header_documents: str
    no_sources_cited: str
    unresolved_reference: str
    no_speakers: str
    generated: str
    coverage_partial: str
    coverage_all: str
    rounds_many: str
    rounds_single: str
    scrubber: str
    scrubber_enabled: str
    scrubber_disabled: str
    mode_names: Dict[str, str]
    # Singular and plural of the counted nouns, for templates that agree with a count ("1 source", "2 sources").
    nouns: Dict[str, Tuple[str, str]] = Field(default_factory=dict)

    def noun(self, key: str, count: int) -> str:
        singular, plural = self.nouns.get(key, ("", ""))
        return singular if count == 1 else plural


class DocumentRules(_Frozen):
    """How documents in this language are located and split: citation labels, clause headings, and OCR."""
    # "p. 7" / "pp. 7–8": the page label a document citation carries.
    page_label: str
    pages_label: str
    # "line 12" / "lines 12–40": what a plain-text or Markdown citation carries -- the file's own line numbers.
    line_label: str
    lines_label: str
    # "image": an image citation's location. The file itself is the whole source; there is nothing finer to point at.
    image_label: str
    # Joins the headings of a section path: "Article 7 Termination › 7.2".
    section_separator: str = " › "
    # A line matching one of these starts a new clause ("Article 7", "3. člen") or a book's chapter ("Chapter 4",
    # "Glava II"), whatever its formatting.
    clause_patterns: List[str] = Field(default_factory=list)
    # RapidOCR recognition model for scanned pages ("latin"). None: this language has no trustworthy OCR model,
    # so scanned pages are reported as unreadable instead of being indexed with text the model guessed.
    ocr_rec_model: Optional[str] = None

    @field_validator("clause_patterns")
    @classmethod
    def _patterns_compile(cls, patterns: List[str]) -> List[str]:
        for pattern in patterns:
            _compiles(pattern)
        return patterns


class ImageRules(_Frozen):
    """
    The labels of an image's citable excerpt. An image has no words of its own, so its excerpt is written by the
    vision model and OCR; these labels say so to the reader and to the answer model, in the appliance's language.
    """
    # Before the vision model's description: "Image (machine description)"
    description_label: str
    # Before the text OCR read off the image
    ocr_text_label: str
    # Before text only the vision model read, on an appliance without OCR to check it against
    unverified_text_label: str
    # Said when OCR's low-confidence reading (handwriting) was left out: the image has text the index doesn't hold
    withheld_text_note: str


class LanguagePack(_Frozen):
    code: str = Field(pattern=r"^[a-z]{2,3}$")
    # English name, used inside English-language prompts ("Write ... in Serbian").
    name: str
    # The language's name for itself, shown where a person picks a language ("Srpski").
    native_name: str
    transcription: TranscriptionRules
    punctuation: PunctuationRules
    search: SearchRules
    text: TextRules
    phonetic: PhoneticRules
    glossary: GlossaryRules
    answers: AnswerRules
    report: ReportStrings
    documents: DocumentRules
    images: ImageRules
    # Settings fields this language overrides (model paths, VRAM budgets, context sizes). Applied by
    # `app.core.config.Settings`, which is where they are read; kept here so a pack validates as a whole.
    models: Dict[str, object] = Field(default_factory=dict)

    def localize_prompt(self, prompt: str) -> str:
        """
        A prompt whose output is stored or shown, adapted to this language: the refusal sentence becomes this
        language's, and the output instruction is appended. English prompts come back unchanged.
        """
        localized = prompt.replace(ENGLISH_REFUSAL, self.answers.refusal)
        if self.answers.output_instruction:
            localized += f"\n\n<OUTPUT_LANGUAGE>\n{self.answers.output_instruction}\n</OUTPUT_LANGUAGE>"
        return localized

    def localize_transcript_prompt(self, prompt: str) -> str:
        """A prompt that edits a transcript in place: it must keep the transcript's language, never translate."""
        if not self.answers.transcript_instruction:
            return prompt
        return f"{prompt}\n\n<TRANSCRIPT_LANGUAGE>\n{self.answers.transcript_instruction}\n</TRANSCRIPT_LANGUAGE>"

    def is_refusal(self, text: str) -> bool:
        """
        True when an answer is the grounded refusal, in this language or in English: a model told to answer in
        Serbian can still fall back to the English sentence it saw in its instructions.
        """
        stripped = text.strip()
        return any(stripped.startswith(sentence.rstrip("."))
                   for sentence in {self.answers.refusal, ENGLISH_REFUSAL})
