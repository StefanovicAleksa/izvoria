# Path: app/features/source_profile/domain/entities.py

import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.features.source_profile.domain.enums import ProfileOrigin

# A title is a name, not a summary; a topic is one or two sentences
MAX_TITLE_CHARS = 120
MAX_TOPIC_CHARS = 400


class SourceIdentity(BaseModel):
    """What the database knows about a source before anyone read it: its name, file and language."""
    model_config = ConfigDict(frozen=True)

    source_id: uuid.UUID
    name: str
    file_name: str = ""
    language: Optional[str] = None


class ProfileDraft(BaseModel):
    """A title and topic as a model (or the file name) proposes them, before they are stored."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=MAX_TITLE_CHARS)
    topic: str = Field(default="", max_length=MAX_TOPIC_CHARS)

    @field_validator("title", "topic", mode="before")
    @classmethod
    def _one_line(cls, value):
        """Model output may wrap; a title or topic is shown on one line."""
        return " ".join(str(value).split()) if value is not None else value


class SourceProfile(BaseModel):
    """
    What a source is about, in words a person would use to refer to it: "Statistics 110, Lecture 4" and "Conditional
    probability and Bayes' rule". It is search material and a label, never citable text.
    """
    model_config = ConfigDict(frozen=True)

    source_id: uuid.UUID
    title: str = Field(..., min_length=1, max_length=MAX_TITLE_CHARS)
    topic: str = Field(default="", max_length=MAX_TOPIC_CHARS)
    origin: ProfileOrigin
    language: Optional[str] = None


class ProfiledSource(BaseModel):
    """A source as scope resolution sees it: its own names, and its profile when it has one."""
    model_config = ConfigDict(frozen=True)

    identity: SourceIdentity
    profile: Optional[SourceProfile] = None

    @property
    def source_id(self) -> uuid.UUID:
        return self.identity.source_id

    @property
    def display_title(self) -> str:
        return self.profile.title if self.profile else self.identity.name


class SourceScope(BaseModel):
    """The one source a question explicitly refers to ("in lecture 4 of Stat 110")."""
    model_config = ConfigDict(frozen=True)

    source_id: uuid.UUID
    title: str
