"""
Handing a cited source to the tester's own browser, when the app runs on the hosted demo box.

There, the box's own player is useless: the tester watches the desktop through noVNC, which carries no sound,
and can't save or print what opens on it. The box's agent launches this app with an address and a token for
this one launch in its environment; "Open source" then posts the citation to the agent, which serves the file to
the browser page beside the desktop. Without those two variables -- the on-premise appliance -- this is never
used, and sources open on the desktop as they always have.

Standard library only: the appliance ships air-gapped, and one small local POST doesn't justify a dependency.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Callable, Mapping, Optional

from izvoria_desktop.domain.entities import Evidence
from izvoria_desktop.utils.formatting import location
from izvoria_desktop.utils.i18n import tr

logger = logging.getLogger(__name__)

OFFER_URL_ENV = "ONYX_SOURCE_OFFER_URL"
OFFER_TOKEN_ENV = "ONYX_SOURCE_OFFER_TOKEN"
TOKEN_HEADER = "X-Offer-Token"
TIMEOUT_SECONDS = 2.0


class SourceOfferClient:
    def __init__(self, url: str, token: str, opener: Callable = urllib.request.urlopen,
                 timeout: float = TIMEOUT_SECONDS) -> None:
        self._url = url
        self._token = token
        self._open = opener
        self._timeout = timeout

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["SourceOfferClient"]:
        url, token = environ.get(OFFER_URL_ENV, ""), environ.get(OFFER_TOKEN_ENV, "")
        return cls(url, token) if url and token else None

    @staticmethod
    def payload(evidence: Evidence) -> dict:
        return {
            "path": evidence.clip_path,
            "media_kind": evidence.media_kind.value,
            "anchor_kind": evidence.anchor_kind,
            "start_sec": evidence.start_sec,
            "end_sec": evidence.end_sec,
            "page_start": evidence.page_start,
            "page_end": evidence.page_end,
            "doc_label": evidence.doc_label,
            "location": location(evidence),
            "excerpt": evidence.excerpt,
            "speakers": list(evidence.speakers),
        }

    def offer(self, evidence: Evidence) -> Optional[str]:
        """
        Sends the citation to the browser. Returns None on success, or a sentence saying why it didn't go -- never
        raises, because a citation that cannot be checked has to say so.
        """
        request = urllib.request.Request(
            self._url,
            data=json.dumps(self.payload(evidence)).encode("utf-8"),
            headers={"Content-Type": "application/json", TOKEN_HEADER: self._token},
            method="POST",
        )
        try:
            with self._open(request, timeout=self._timeout) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            logger.warning(f"The box refused to offer {evidence.doc_label}: HTTP {exc.code}")
            if exc.code == 400:
                return tr("This source can't be sent to your browser: the appliance doesn't serve that file.")
            return tr("This source couldn't be sent to your browser (the appliance refused, error {code}).",
                      code=exc.code)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning(f"Could not reach the box's agent to offer {evidence.doc_label}: {exc}")
            return tr("This source couldn't be sent to your browser: the appliance's file service isn't answering.")
        logger.info(f"Offered {evidence.doc_label} to the browser.")
        return None
