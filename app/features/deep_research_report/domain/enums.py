# Path: app/features/deep_research_report/domain/enums.py

from enum import Enum


class ReportMode(str, Enum):
    """
    How hard a report looks before it writes.

    DEEP gathers once: decompose the topic, search, write. Fast and usually enough.

    EXHAUSTIVE repeats gathering while sections come back unsupported, because a section with no
    evidence does not distinguish "the corpus does not cover this" from "the sub-question was worded
    badly". Each extra round costs one embedder load, one reranker load and one reload of the report
    model, so it is a separate mode rather than the default.
    """
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class ReportStatus(str, Enum):
    """
    Where a report is in its lifecycle. Every transition is committed, so a crash or a cancellation
    leaves a row that says exactly how far the run got and what can be resumed.
    """
    AWAITING_APPROVAL = "awaiting_approval"   # scoping done, the user has not accepted the outline yet
    GATHERING = "gathering"                   # sweeping the corpus into the evidence bank
    WRITING = "writing"                       # bank is numbered and persisted; sections are being written
    COMPLETED = "completed"
    CANCELLED = "cancelled"                   # stopped by the user; written sections are kept and resumable
    FAILED = "failed"


class SectionStatus(str, Enum):
    PENDING = "pending"
    WRITTEN = "written"
    # The outline proposed it but no evidence in the corpus supported it. Listed under "Not covered by the
    # available material" rather than written from the model's own knowledge.
    SKIPPED_NO_EVIDENCE = "skipped_no_evidence"


class EvidenceOrigin(str, Enum):
    """How a chunk reached the evidence bank."""
    FAN_OUT = "fan_out"   # matched one of the plan's sub-questions
    SWEEP = "sweep"       # surfaced by the per-document sweep, which no sub-question had asked for
    NEIGHBOR = "neighbor"  # pulled in as adjacent transcript context for a chunk that did match
