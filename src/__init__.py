"""Пакет прогнозирования золото-урановых рудных узлов методами ML.

Единственный источник логики проекта. Запуск пайплайна — через
``notebook/forecast_main.ipynb``. Все настройки собраны в :mod:`src.config`.
"""

__version__ = "1.0.0"


def _fix_proj_data() -> None:
    """Снять чужую базу PROJ с пути до первого обращения rasterio к CRS.

    Установщик PostGIS прописывает в систему ``PROJ_LIB`` со своей базой
    (PROJ 6-й раскладки). rasterio берёт её вместо собственной и падает на
    любом ``CRS.from_epsg`` сообщением ``DATABASE.LAYOUT.VERSION.MINOR = 2``.
    Переменная окружения — свойство машины, а не проекта, поэтому чинить её
    приходится здесь: пакет импортируется раньше любого вызова rasterio.
    """
    import importlib.util
    import os
    from pathlib import Path

    # Путь ищем через find_spec: сам ``import rasterio`` уже запоминает базу
    # PROJ, и правка переменной после него ничего не меняет.
    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.origin:
        return
    own = Path(spec.origin).parent / "proj_data"
    if not (own / "proj.db").exists():
        return
    for var in ("PROJ_LIB", "PROJ_DATA"):
        cur = os.environ.get(var)
        if cur and Path(cur).resolve() != own.resolve():
            os.environ[var] = str(own)


_fix_proj_data()
