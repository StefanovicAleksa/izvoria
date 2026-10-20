"""The deployed version a copy of the app reports."""

from web_deploy.common.release import read_release


class TestReadRelease:
    def test_the_release_file_names_the_commit(self, tmp_path):
        (tmp_path / "RELEASE").write_text("9f58193abcde\n")
        assert read_release(tmp_path) == "9f58193abcde"

    def test_a_copy_never_deployed_is_unknown(self, tmp_path):
        assert read_release(tmp_path) == "unknown"

    def test_an_empty_file_is_unknown(self, tmp_path):
        (tmp_path / "RELEASE").write_text("\n")
        assert read_release(tmp_path) == "unknown"
