from unittest.mock import patch

from app.core.database.utils.model_registry import import_all_models


class TestImportAllModels:
    def test_imports_every_feature_models_module_by_dotted_name(self, tmp_path):
        for relative in ("app/features/b/data/models.py", "app/core/a/data/models.py", "app/features/b/data/other.py"):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        with patch("app.core.database.utils.model_registry.settings") as settings, \
                patch("app.core.database.utils.model_registry.importlib.import_module") as import_module:
            settings.BASE_DIR = tmp_path
            import_all_models()

        assert [c.args[0] for c in import_module.call_args_list] == [
            "app.core.a.data.models", "app.features.b.data.models"]
