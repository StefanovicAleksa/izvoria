import json

import httpx
import pytest

from deploy.gateway.vast_client import FakeVastClient, InstanceInfo, VastClient, VastError

URL = "https://console.vast.ai/api/v0/instances/4242/"


def client_with(handler, requests):
    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)
    return VastClient("KEY", 4242, transport=httpx.MockTransport(record))


class TestStartStop:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("method, state", [("start", "running"), ("stop", "stopped")])
    async def test_puts_the_state_with_the_bearer_key(self, method, state):
        requests = []
        client = client_with(lambda r: httpx.Response(200, json={"success": True}), requests)

        await getattr(client, method)()

        (request,) = requests
        assert (request.method, str(request.url)) == ("PUT", URL)
        assert json.loads(request.content) == {"state": state}
        assert request.headers["authorization"] == "Bearer KEY"

    @pytest.mark.asyncio
    async def test_vast_refusal_becomes_a_vast_error_with_its_message(self):
        client = client_with(lambda r: httpx.Response(200, json={"success": False, "msg": "no such"}), [])
        with pytest.raises(VastError, match="no such"):
            await client.start()

    @pytest.mark.asyncio
    async def test_http_failure_becomes_a_vast_error(self):
        client = client_with(lambda r: httpx.Response(500, text="boom"), [])
        with pytest.raises(VastError):
            await client.stop()


class TestInfo:
    @pytest.mark.asyncio
    async def test_reads_status_ip_and_mapped_ports(self):
        row = {
            "actual_status": "running", "intended_status": "running", "public_ipaddr": "203.0.113.9",
            "status_msg": "ok",
            "ports": {"8090/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41001"}],
                      "6080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41002"}],
                      "53/udp": [{"HostPort": "1"}], "22/tcp": []},
        }
        requests = []
        info = await client_with(lambda r: httpx.Response(200, json={"instances": row}), requests).info()

        assert str(requests[0].url) == URL + "?owner=me"
        assert info == InstanceInfo("running", "running", "203.0.113.9", {8090: 41001, 6080: 41002}, "ok")
        assert info.address(8090) == "203.0.113.9:41001"
        assert info.address(9999) is None

    @pytest.mark.asyncio
    async def test_a_destroyed_instance_is_an_error(self):
        client = client_with(lambda r: httpx.Response(200, json={"instances": None}), [])
        with pytest.raises(VastError, match="destroyed"):
            await client.info()


class TestFakeVastClient:
    @pytest.mark.asyncio
    async def test_flips_state_and_maps_ports_to_themselves(self):
        fake = FakeVastClient("127.0.0.1", (8090, 6080))
        assert not (await fake.info()).running
        await fake.start()
        info = await fake.info()
        assert info.running and info.address(6080) == "127.0.0.1:6080"
        await fake.stop()
        assert fake.stops == 1 and not (await fake.info()).running
