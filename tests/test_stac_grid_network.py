"""Сетевая устойчивость чтения растров (src/stac_grid.py).

Оба контракта здесь появились из измеренных отказов прогона 26.08.2026, и оба
про одно: сбой обязан становиться ОШИБКОЙ, а не тишиной.

* поиск в каталоге отдавал 504 и «request exceeded the maximum allowed time», и
  три сезонных окна пропали целиком, хотя данные были на месте — у ``search``
  не было повторов, хотя у ``read_on_grid`` рядом они были;
* чтение сцены висело бесконечно: 8 минут при нулевом приросте кэша и почти
  нулевом процессорном времени. Повторы там не помогают по построению — до
  исключения дело не доходит, повторять нечего.
"""
import inspect

from src import config, stac_grid


def test_gdal_env_sets_timeouts_by_default():
    got = stac_grid.gdal_env()
    assert got["GDAL_HTTP_TIMEOUT"] == str(config.SAT_HTTP_TIMEOUT_S)
    assert got["GDAL_HTTP_CONNECTTIMEOUT"] == str(config.SAT_HTTP_CONNECT_TIMEOUT_S)


def test_gdal_env_also_sets_a_speed_floor():
    """Абсолютного таймаута мало: на многопоточной подкачке он не срабатывает.

    Измерено: процесс простоял 16 минут с тремя установленными соединениями и
    нулевым приёмом, хотя тот же блоб читался обычным запросом за 1.2 с. Случай
    «соединение есть, данных нет» ловит только порог скорости.
    """
    got = stac_grid.gdal_env()
    assert got["GDAL_HTTP_LOW_SPEED_LIMIT"] == str(config.SAT_HTTP_LOW_SPEED_LIMIT)
    assert got["GDAL_HTTP_LOW_SPEED_TIME"] == str(config.SAT_HTTP_LOW_SPEED_TIME_S)


def test_speed_floor_is_far_below_any_working_link():
    """Порог обязан быть заведомо ниже рабочей скорости, иначе резал бы живое чтение."""
    assert config.SAT_HTTP_LOW_SPEED_LIMIT <= 10_000        # 10 КБ/с
    assert 10 <= config.SAT_HTTP_LOW_SPEED_TIME_S <= config.SAT_HTTP_TIMEOUT_S


def test_caller_env_wins_over_the_defaults():
    """У ASTER TIR своя авторизация Earthdata — её настройки перебивать нельзя."""
    got = stac_grid.gdal_env({"GDAL_HTTP_TIMEOUT": "5", "GDAL_HTTP_COOKIEFILE": "/tmp/c"})
    assert got["GDAL_HTTP_TIMEOUT"] == "5"
    assert got["GDAL_HTTP_COOKIEFILE"] == "/tmp/c"
    assert got["GDAL_HTTP_CONNECTTIMEOUT"] == str(config.SAT_HTTP_CONNECT_TIMEOUT_S)


def test_gdal_env_values_are_strings():
    """GDAL читает настройки как строки; int молча не сработал бы у части опций."""
    assert all(isinstance(v, str) for v in stac_grid.gdal_env().values())


def test_timeouts_leave_room_for_an_honestly_slow_scene():
    """Таймаут не должен резать честную медленную сцену: она читается десятки секунд."""
    assert config.SAT_HTTP_TIMEOUT_S >= 90
    assert config.SAT_HTTP_CONNECT_TIMEOUT_S < config.SAT_HTTP_TIMEOUT_S


def test_href_on_grid_actually_uses_the_env_helper():
    """Регрессия: без этого таймауты остались бы объявленными, но не применёнными."""
    src = inspect.getsource(stac_grid.href_on_grid)
    assert "rasterio.Env(**gdal_env(env))" in src


def test_catalogue_client_is_built_with_an_explicit_timeout():
    """Повторы у поиска бесполезны, пока сам запрос может висеть без конца.

    Измерено 28.08.2026: поиск сцен одного окна простоял 15 минут, не получив ни
    байта и не подняв исключения, — ``pystac_client`` по умолчанию не ставит
    таймаута вовсе. Дыра та же, что закрыл ``GDAL_HTTP_TIMEOUT`` для растров.
    """
    src = inspect.getsource(stac_grid.catalog)
    assert "stac_io=StacApiIO" in src
    assert "config.SAT_STAC_TIMEOUT_S" in src
    assert "config.SAT_HTTP_CONNECT_TIMEOUT_S" in src


def test_catalogue_timeout_is_tighter_than_the_raster_one():
    """Страница JSON приходит за секунды; терпеть её минутами незачем."""
    assert config.SAT_STAC_TIMEOUT_S < config.SAT_HTTP_TIMEOUT_S
    assert config.SAT_STAC_TIMEOUT_S >= 30      # но и честную медленную страницу не резать


def test_search_retries_the_catalogue_query():
    """Регрессия на 504 от каталога: у поиска обязаны быть те же повторы, что у чтения."""
    src = inspect.getsource(stac_grid.search)
    assert "config.SAT_FETCH_RETRIES" in src
    assert "config.SAT_FETCH_BACKOFF_S" in src
    # Страницы тянутся лениво, поэтому list(res.items()) обязан быть ВНУТРИ try:
    # иначе повтор охватывал бы только сам запрос, а не выборку.
    assert "list(res.items())" in src.split("try:")[1].split("except")[0]
