# Canvas: test_visual_synthesis_fuzzy_grouper_v1
# Path: tests/features/visual_context_synthesis/utils/test_fuzzy_grouper.py

import uuid
from dataclasses import dataclass

from app.features.visual_context_synthesis.utils.fuzzy_grouper import build_fuzzy_visual_timeline, \
    calculate_jaccard_similarity


# Simple mock for FrameAnalysis
@dataclass
class MockFrameAnalysis:
    frame_id: uuid.UUID
    timestamp_sec: float
    description: str


class TestFuzzyGrouper:
    def test_calculate_jaccard_similarity(self):
        # High overlap (stop words stripped)
        text1 = "A financial chart showing a sudden spike."
        text2 = "The financial chart depicts a sudden spike."
        score = calculate_jaccard_similarity(text1, text2)
        assert score >= 0.7  # "financial", "chart", "sudden", "spike" overlap perfectly

        # No overlap
        text3 = "A man pointing at the audience."
        score_diff = calculate_jaccard_similarity(text1, text3)
        assert score_diff == 0.0

    def test_build_timeline_groups_fuzzy_matches(self):
        id1, id2, id3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        analyses = [
            MockFrameAnalysis(id1, 10.0, "A line chart showing a spike."),
            MockFrameAnalysis(id2, 12.0, "The line chart depicts a spike."),  # Fuzzy match to 1
            MockFrameAnalysis(id3, 20.0, "A doctor pointing at an X-Ray.")  # Complete change
        ]

        accepted_ids = {id1, id2, id3}

        timeline = build_fuzzy_visual_timeline(analyses, accepted_ids, similarity_threshold=0.65)

        # Should be 2 events
        assert len(timeline) == 2

        # Event 1 spans 10.0 to 12.0
        assert timeline[0]["start_sec"] == 10.0
        assert timeline[0]["end_sec"] == 12.0
        assert len(timeline[0]["frame_ids"]) == 2

        # Event 2 starts at 20.0
        assert timeline[1]["start_sec"] == 20.0
        assert len(timeline[1]["frame_ids"]) == 1

    def test_build_timeline_filters_rejected_frames(self):
        id1, rejected_id = uuid.uuid4(), uuid.uuid4()

        analyses = [
            MockFrameAnalysis(id1, 10.0, "Valid chart."),
            MockFrameAnalysis(rejected_id, 15.0, "Hallucinated cat.")
        ]

        # Only id1 survived the Gatekeeper
        accepted_ids = {id1}

        timeline = build_fuzzy_visual_timeline(analyses, accepted_ids)

        assert len(timeline) == 1
        assert timeline[0]["frame_ids"] == [id1]
        assert timeline[0]["representative_description"] == "Valid chart."