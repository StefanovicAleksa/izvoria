from pathlib import Path

from web_deploy.local.run_local import EXAMPLE, build_environment, commands, ensure_env_file, parse_env_file


class TestLocalEnvironment:
    def test_the_first_run_writes_a_private_env_with_a_fresh_secret(self, tmp_path):
        env_file = tmp_path / "local.env"
        assert ensure_env_file(env_file, EXAMPLE) is True
        assert ensure_env_file(env_file, EXAMPLE) is False
        values = parse_env_file(env_file)
        assert len(values["WEB_API_SECRET"]) == 64
        assert oct(env_file.stat().st_mode)[-3:] == "600"

    def test_the_example_selects_dev_hardware_behaviour(self):
        values = parse_env_file(EXAMPLE)
        assert (values["BOX_MODE"], values["GATEWAY_HTTPS_ONLY"], values["GPU_SHARING"]) == ("local", "0", "exclusive")
        assert "FIRM_ROUTING" not in values  # every firm signs in at one address now
        assert values["LLM_SERVER_KV_TOKENS"] == "131072"  # the answer model keeps its full context
        assert values["MODEL_PINNED_TYPES"] == '["rag_embedding","rag_reranking"]'  # quotes removed, JSON kept

    def test_paths_are_derived_under_the_data_dir(self, tmp_path):
        env = build_environment({"LOCAL_DATA_DIR": "data/web", "LOCAL_PORT": "9000"}, {"PATH": "/bin"}, root=tmp_path)
        assert env["TENANTS_ROOT"] == str(tmp_path / "data/web/tenants")
        assert env["CONTROL_DATABASE_URL"] == f"sqlite:///{tmp_path / 'data/web/control.sqlite3'}"
        assert env["LLAMA_SERVER_BIN"] == str(tmp_path / "data/web/llama.cpp/build/bin/llama-server")
        assert (env["ENV"], env["PATH"], env["WEB_API_URL"]) == ("development", "/bin", "http://127.0.0.1:8095")

    def test_the_api_stays_local_and_the_gateway_faces_the_lan(self, tmp_path):
        cmds = commands({"LOCAL_PORT": "9000"}, python="py")
        assert cmds["api"][cmds["api"].index("--host") + 1] == "127.0.0.1"
        assert cmds["gateway"][cmds["gateway"].index("--host") + 1] == "0.0.0.0"
        assert cmds["gateway"][-1] == "9000"
        assert cmds["ingest"] == ["py", "-m", "web_deploy.ingest_worker.supervisor"]
