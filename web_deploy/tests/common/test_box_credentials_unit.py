import pytest

from app.core.database.data import credentials
from web_deploy.common.box_credentials import CredentialsUnavailable, install_from_env, password_provider


@pytest.fixture(autouse=True)
def reset():
    credentials.set_password_provider(None)
    yield
    credentials.set_password_provider(None)


class TestPasswordProvider:
    def test_asks_the_gateway_signed_and_caches_briefly(self):
        now = [0.0]
        asked = []

        def fetch(url, headers):
            asked.append((url, sorted(headers)))
            return f"pw{len(asked)}"

        current = password_provider("http://127.0.0.1:18000/internal/box/credentials", "k" * 40, cache_seconds=10,
                                    fetch=fetch, clock=lambda: now[0], sleep=lambda s: None)
        assert current() == "pw1"
        assert current() == "pw1"
        now[0] = 11
        assert current() == "pw2"
        assert asked[0] == ("http://127.0.0.1:18000/internal/box/credentials", ["x-onyx-service-sig", "x-onyx-service-ts"])

    def test_waits_for_the_gateway_then_gives_up(self):
        now = [0.0]
        answers = iter([CredentialsUnavailable("starting"), "pw"])

        def fetch(url, headers):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        def sleep(seconds):
            now[0] += seconds

        assert password_provider("http://x/p", "s", fetch=fetch, clock=lambda: now[0], sleep=sleep)() == "pw"

        def never(url, headers):
            raise CredentialsUnavailable("The box is not started.")

        with pytest.raises(CredentialsUnavailable):
            password_provider("http://x/p", "s", wait_seconds=5, fetch=never, clock=lambda: now[0], sleep=sleep)()


class TestInstall:
    def test_only_on_the_hosted_box(self):
        assert install_from_env({}) is False
        assert credentials.current_password() is None
        assert install_from_env({"BOX_CREDENTIALS_URL": "http://127.0.0.1:18000/internal/box/credentials",
                                 "WEB_API_SECRET": "k" * 40}) is True
        assert credentials._provider is not None
