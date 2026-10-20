# Path: tests/features/source_profile/utils/test_scope_matcher.py

import uuid

from app.features.source_profile.domain.entities import ProfiledSource, SourceIdentity, SourceProfile
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.utils.scope_matcher import shortlist, tokens


def _source(name: str, title: str = None) -> ProfiledSource:
    source_id = uuid.uuid4()
    profile = SourceProfile(source_id=source_id, title=title, origin=ProfileOrigin.GENERATED) if title else None
    return ProfiledSource(identity=SourceIdentity(source_id=source_id, name=name, file_name=name), profile=profile)


def _stat110():
    return [_source(f"Lecture {n}: Topic {n} | Statistics 110 [abcdefghijk].mp4") for n in range(1, 31)]


class TestTokens:
    def test_numbers_lose_leading_zeros_and_underscores_split_words(self):
        assert tokens("module_01.mp4") == {"module", "1", "mp4"}

    def test_unicode_words_are_kept(self):
        assert tokens("Predavanje četiri") == {"predavanje", "četiri"}


class TestShortlist:
    def test_a_numbered_lecture_is_ranked_first_among_its_series(self):
        sources = _stat110()
        result = shortlist("what are all the formulas mentioned in lecture 4 of stat 110", sources)
        assert result[0] is sources[3]

    def test_lecture_4_is_not_lecture_14(self):
        sources = _stat110()
        result = shortlist("lecture 14 of stat 110", sources)
        assert result[0] is sources[13]
        assert sources[3] not in result[:1]

    def test_a_topic_question_names_no_source(self):
        assert shortlist("what is conditional probability", _stat110()) == []

    def test_short_words_never_list_a_source(self):
        sources = [_source("History of Rome.pdf"), _source("Contract.pdf"), _source("Invoice.pdf")]
        assert shortlist("what is the meaning of life", sources) == []

    def test_a_generated_title_names_a_badly_named_file(self):
        module = _source("module_01.mp4", title="Introduction to Contract Law, Module 1")
        other = _source("module_02.mp4", title="Torts, Module 2")
        assert shortlist("summarise the contract law module", [module, other]) == [module]

    def test_an_abbreviation_matches_the_full_word(self):
        sources = [_source("Statistics Lecture.mp4"), _source("Biology Lecture.mp4")]
        assert shortlist("what did the stat lecture cover", sources) == [sources[0]]

    def test_no_sources_or_no_words(self):
        assert shortlist("lecture 4", []) == []
        assert shortlist("?!", _stat110()) == []

    def test_at_most_limit_candidates(self):
        sources = [_source(f"Deposition {n}.pdf") for n in range(10)] + [_source(f"Other {n}.pdf") for n in range(12)]
        assert len(shortlist("deposition", sources, limit=3)) == 3
