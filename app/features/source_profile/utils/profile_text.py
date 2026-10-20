# Path: app/features/source_profile/utils/profile_text.py

"""Titles and topics cut to size, and a readable title from a bare file name."""

import os
import re

from app.features.source_profile.domain.entities import MAX_TITLE_CHARS, MAX_TOPIC_CHARS, ProfileDraft

# "<title> [<11-char YouTube id>]": the name a YouTube download is saved under (web_deploy/common/youtube.py)
_YOUTUBE_ID_SUFFIX = re.compile(r"\s*\[[A-Za-z0-9_-]{11}\]\s*$")
_SEPARATORS = re.compile(r"[_]+|(?<=\w)\.(?=\w)")
# A file extension: a dot and one to five letters or digits, at least one a letter (".mp4", not ".5" of "v2.5")
_EXTENSION = re.compile(r"\.(?=[A-Za-z0-9]{0,4}[A-Za-z])[A-Za-z0-9]{1,5}")


def title_from_name(name: str) -> str:
    """A readable title from a source's name: no extension, no YouTube id, underscores as spaces."""
    title = " ".join(_SEPARATORS.sub(" ", _strip_file_parts(name)).split())
    return clip(title or name.strip() or "Untitled", MAX_TITLE_CHARS)


def _strip_file_parts(name: str) -> str:
    stem, extension = os.path.splitext(name.strip())
    if not extension or not _EXTENSION.fullmatch(extension):
        stem = name.strip()
    return _YOUTUBE_ID_SUFFIX.sub("", stem)


def clip(text: str, limit: int) -> str:
    """`text` cut to `limit` characters at a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-–—")
    return cut or text[:limit]


def clipped_draft(title: str, topic: str) -> ProfileDraft:
    """
    A draft whose fields fit their limits; a model that writes too much is cut, not rejected. A title copied from
    the file name loses the file's extension and YouTube id.
    """
    tidy = " ".join(_strip_file_parts(title).split()) or title
    return ProfileDraft(title=clip(tidy, MAX_TITLE_CHARS), topic=clip(topic, MAX_TOPIC_CHARS))


def first_sentence(text: str, limit: int = MAX_TOPIC_CHARS) -> str:
    text = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    return clip(match.group(1) if match else text, limit)

