# Path: scripts/eval/cross_language/labels.py

"""
Question sets for one (material language, question language) pair, in rag_eval's label format.

The questions come from the language benchmark's labels.json of the material's language, translated into every
other language (translations.json). The relevant chunks don't change with the question's language, which is what
makes the pairs comparable: the same question, about the same chunk, asked in another language.
"""

from typing import Dict, List

# Unanswerable questions are asked once per question language, with that language's native set
NATIVE_ONLY = "unanswerable"


def pair_labels(translations: dict, data: str, query: str, unanswerable: Dict[str, List[str]],
                cyrillic=None) -> dict:
    """
    `translations[data]["answerable"]` items carry `text: {code: question}`. `cyrillic`, when given, converts the
    question-language text to Cyrillic, for Serbian typed in Cyrillic ("<id>c" ids).
    """
    answerable = [{"id": item["id"], "question": item["text"][query], "relevant": item["relevant"],
                   "source": item.get("source", "")} for item in translations[data]["answerable"]]
    if cyrillic is not None:
        answerable += [{**item, "id": f"{item['id']}c", "question": cyrillic(item["question"])}
                       for item in list(answerable)]
    return {
        "_comment": f"{data} material, questions in {query}{' (Latin and Cyrillic)' if cyrillic else ''}.",
        "answerable": answerable,
        "unanswerable": [{"id": f"u{i:02d}", "question": q}
                         for i, q in enumerate(unanswerable[query], 1)] if data == query else [],
    }
