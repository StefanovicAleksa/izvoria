# Path: app/core/database/utils/model_registry.py

import importlib

from app.core.config import settings


def import_all_models() -> None:
    """
    Imports every `app/**/data/models.py`, so `Base.metadata` holds every table. A table is otherwise known only
    if something happened to import its model, and a migration diffed against a partial metadata would drop the
    rest.
    """
    for models_file in sorted(settings.BASE_DIR.glob("app/**/data/models.py")):
        importlib.import_module(".".join(models_file.relative_to(settings.BASE_DIR).with_suffix("").parts))
