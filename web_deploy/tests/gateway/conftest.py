import httpx
import pytest

from deploy.gateway.vast_client import FakeVastClient
from web_deploy.gateway.app import create_app
from web_deploy.gateway.box import BoxController
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.idle import IdleMonitor
from web_deploy.gateway.mailer import Mailer, MailSettings
from web_deploy.gateway.store import OWNER_FIRM, Store
from web_deploy.tests.gateway.support import (
    CONSOLE_PASSWORD, FIRM_PASSWORD, OTHER_FIRM_PASSWORD, PASSWORD, SECRET, BoxApi, InlineExecutor,
    RecordingTransport,
)

MAIL = MailSettings(smtp_host="smtp.test", mail_from="onyx@onyx.test", signup_to="sales@onyx.test",
                    support_to="help@onyx.test")


@pytest.fixture
def box_api():
    return BoxApi()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "g.sqlite3")
    s.create_firm("legal-demo", "Legal demo", "en", is_demo=True, password=FIRM_PASSWORD)
    s.create_firm("other-firm", "Other firm", "sr", password=OTHER_FIRM_PASSWORD)
    s.create_user("legal-demo", "alice@firm.com", PASSWORD, "member")
    s.create_user("legal-demo", "boss@firm.com", PASSWORD, "admin")
    s.create_user("other-firm", "eve@other.com", PASSWORD, "admin")
    s.create_user(OWNER_FIRM, "owner@onyx.test", PASSWORD, "owner")
    s.set_firm_password(OWNER_FIRM, CONSOLE_PASSWORD)
    return s


@pytest.fixture
def vast():
    return FakeVastClient("127.0.0.1", (8095,))


@pytest.fixture
def transport():
    return RecordingTransport()


@pytest.fixture
def app(tmp_path, store, vast, box_api, transport):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>spa</html>")
    settings = GatewaySettings(api_secret=SECRET, database_path=tmp_path / "g.db", static_dir=static,
                               youtube_cookies_file=tmp_path / "youtube_cookies.txt",
                               fake_box_host="127.0.0.1", https_only=False, mail=MAIL)
    http = httpx.AsyncClient(transport=httpx.MockTransport(box_api))
    box = BoxController(settings, vast, http)
    mailer = Mailer(MAIL, store, transport=transport, executor=InlineExecutor())
    return create_app(settings, store, box, IdleMonitor(box, 1800), http, run_idle_monitor=False, mailer=mailer,
                      glossary_root=tmp_path / "tenants")
