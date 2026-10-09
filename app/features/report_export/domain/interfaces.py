# Path: app/features/report_export/domain/interfaces.py

from abc import ABC, abstractmethod

from app.features.deep_research_report.domain.entities import Report


class IReportDocumentWriter(ABC):
    """Contract for turning a finished (or partial) report into one document file, as bytes."""

    @abstractmethod
    def write(self, report: Report) -> bytes:
        """
        The whole report: title, provenance, summary, sections, what was not covered, and references that every
        `[Doc N]` resolves to. Made from the report as stored, never from a model.
        """
        pass
