"""
On the hosted demo box, "Open source" posts the citation to the box's agent so it reaches the tester's browser.
Never a real request here: the opener is a mock.
"""

import json
import urllib.error
import uuid
from unittest.mock import MagicMock

import pytest

from izvoria_desktop.data.source_offer_client import (
    OFFER_TOKEN_ENV, OFFER_URL_ENV, TOKEN_HEADER, SourceOfferClient,
)
from izvoria_desktop.domain.entities import Evidence
from izvoria_desktop.domain.enums import MediaKind

URL = "http://127.0.0.1:8090/offers"


def make_evidence(**overrides):
    values = dict(
        doc_number=3, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), start_sec=2001.0, end_sec=2093.0,
        excerpt="SPEAKER_01: there are some very key reference points.", speakers=("SPEAKER_01",),
        clip_path="/workspace/onyx_v3/onyx_data/2026/10/03/a.mp3", media_kind=MediaKind.FULL_SOURCE,
        anchor_kind="time",
    )
    values.update(overrides)
    return Evidence(**values)


def opener_returning(status=201):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"id": "x"}'
    return MagicMock(return_value=response)


class TestFromEnv:
    @pytest.mark.parametrize("environ", [{}, {OFFER_URL_ENV: URL}, {OFFER_TOKEN_ENV: "t"},
                                         {OFFER_URL_ENV: "", OFFER_TOKEN_ENV: "t"}])
    def test_the_on_premise_appliance_has_no_client(self, environ):
        assert SourceOfferClient.from_env(environ) is None

    def test_the_hosted_box_has_one(self):
        assert SourceOfferClient.from_env({OFFER_URL_ENV: URL, OFFER_TOKEN_ENV: "t"}) is not None


class TestOffer:
    def test_posts_the_path_and_anchor_with_the_launch_token(self):
        opener = opener_returning()
        problem = SourceOfferClient(URL, "tok", opener=opener).offer(make_evidence())

        assert problem is None
        request = opener.call_args.args[0]
        assert opener.call_args.kwargs == {"timeout": 2.0}
        assert (request.full_url, request.get_method()) == (URL, "POST")
        assert request.get_header(TOKEN_HEADER.capitalize()) == "tok"
        assert request.get_header("Content-type") == "application/json"
        assert json.loads(request.data) == {
            "path": "/workspace/onyx_v3/onyx_data/2026/10/03/a.mp3",
            "media_kind": "full_source",
            "anchor_kind": "time",
            "start_sec": 2001.0,
            "end_sec": 2093.0,
            "page_start": None,
            "page_end": None,
            "doc_label": "[Doc 3]",
            "location": "33:21–34:53",
            "excerpt": "SPEAKER_01: there are some very key reference points.",
            "speakers": ["SPEAKER_01"],
        }

    def test_a_document_sends_its_page(self):
        opener = opener_returning()
        evidence = make_evidence(media_kind=MediaKind.DOCUMENT, anchor_kind="page", page_start=7, page_end=8,
                                 start_sec=0.0, end_sec=0.0, location="pp. 7–8 · Article 7")
        SourceOfferClient(URL, "tok", opener=opener).offer(evidence)

        body = json.loads(opener.call_args.args[0].data)
        assert (body["anchor_kind"], body["page_start"], body["page_end"]) == ("page", 7, 8)
        assert body["location"] == "pp. 7–8 · Article 7"

    @pytest.mark.parametrize("code, words", [(400, "doesn't serve that file"), (401, "error 401"),
                                             (500, "error 500")])
    def test_a_refusal_is_a_sentence_not_an_exception(self, code, words):
        opener = MagicMock(side_effect=urllib.error.HTTPError(URL, code, "no", {}, None))
        problem = SourceOfferClient(URL, "tok", opener=opener).offer(make_evidence())
        assert words in problem

    @pytest.mark.parametrize("error", [urllib.error.URLError("refused"), TimeoutError("slow"),
                                       ConnectionResetError()])
    def test_an_unreachable_agent_is_a_sentence_not_an_exception(self, error):
        opener = MagicMock(side_effect=error)
        problem = SourceOfferClient(URL, "tok", opener=opener).offer(make_evidence())
        assert "isn't answering" in problem
