"""Открытые слои на широкую территорию, которых в проекте ещё не было.

Чего здесь НЕТ: точек рудопроявлений, геохимии и аэрогамма-спектрометрии —
они проверены и закрыты отрицательно (``docs II presentation/
Внешние-источники-золото-проверено.md``, реестр ``meta.gap``). Здесь — слои,
которые закрывают дыры покрытия и ограничения, найденные уже в работе:

* **глобальная геофизика GMT** (``datacache/world_geo``) — EMAG2 магнитка 4 км,
  гравика в свободном воздухе, вертикальный градиент, геоид. Восстановление
  прежнего кэша (``experiments/common.py``); на широкой сетке это региональный
  слой там, где кончается съёмка организации (``gm_*`` 88%, ``pf_*`` 84%);
* **гидросеть OpenStreetMap** (``datacache/osm_hydro``) — ``dist_dnl``/
  ``dist_dnara`` выброшены из набора переноса, потому что реки организации
  оцифрованы только на листе и за его рамкой признак меряет расстояние до края
  оцифровки. OSM покрывает всю широкую сетку одним источником;
* **ESA WorldCover 2021, 10 м** (``datacache/worldcover``) — классы покрова,
  в том числе «голый грунт / редкая растительность»: независимая от Sentinel-2
  маска обнажённости (нынешняя строилась по NDVI того же Sentinel-2);
* **ASTER GED AG100 v3** (``datacache/aster_ged``) — средняя излучательная
  способность по всему архиву ASTER, 100 м. Тепловой канал чувствителен к
  кремнезёму, а средняя по архиву не зависит от одной сцены — то, чего не
  хватает мозаике ``ast_*``. Нужна учётная запись Earthdata (``~/_netrc``).

Только скачивание, без признаков: что из этого идёт в датасет, решает сверка.
Идемпотентно. Запуск из корня: ``python -u -m experiments.fetch_open_layers``.
"""
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cache_paths import CACHE  # noqa: E402
from experiments import common  # noqa: E402

#: Охват широкой сетки с запасом, градусы (103.93-108.10 в.д., 69.98-71.36 с.ш.).
BBOX = (103.8, 69.9, 108.3, 71.45)   # (запад, юг, восток, север)

OVERPASS = ("https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def curl(url: str, out: Path, netrc: bool = False) -> bool:
    """Скачать с докачкой; ``-n`` подставляет логин Earthdata из ``~/_netrc``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1000:
        return True
    cmd = ["curl", "-sS", "-L", "--fail", "--connect-timeout", "30", "--max-time", "1800",
           "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(out), url]
    if netrc:
        cookies = str(out.parent / ".cookies")
        cmd[1:1] = ["-n", "-b", cookies, "-c", cookies]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        out.unlink(missing_ok=True)
        log(f"  НЕ СКАЧАН {url}: {r.stderr.strip()[:200]}")
        return False
    return True


def world_geo() -> None:
    tile, _, _ = common.tile_name(BBOX[0], BBOX[1])
    for ds in common.DS:
        p = common.fetch(ds, tile)
        log(f"  GMT {ds} {tile}: {'есть' if p else 'НЕ СКАЧАН'}")


def osm_hydro() -> None:
    out = CACHE / "osm_hydro" / "waterways.json"
    if out.exists():
        log(f"  OSM: уже есть ({out.stat().st_size / 1e6:.1f} МБ)")
        return
    s, w, n, e = BBOX[1], BBOX[0], BBOX[3], BBOX[2]
    q = (f"[out:json][timeout:900];("
         f"way[waterway~\"^(river|stream|canal)$\"]({s},{w},{n},{e});"
         f"way[natural=water]({s},{w},{n},{e});"
         f"relation[natural=water]({s},{w},{n},{e}););out geom;")
    data = None
    for server in OVERPASS:
        req = urllib.request.Request(server, data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": "GIS-project/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=1200) as resp:
                data = resp.read()
            break
        except Exception as exc:  # noqa: BLE001 — публичные серверы Overpass часто перегружены
            log(f"  OSM {server}: {exc!r}")
    if data is None:
        raise RuntimeError("ни один сервер Overpass не ответил")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    els = json.loads(data).get("elements", [])
    kinds: dict[str, int] = {}
    for el in els:
        t = el.get("tags", {})
        k = t.get("waterway") or t.get("natural") or "?"
        kinds[k] = kinds.get(k, 0) + 1
    log(f"  OSM: {len(els)} объектов {kinds}, {len(data) / 1e6:.1f} МБ")


def worldcover() -> None:
    # Тайлы 3x3 градуса, имя по юго-западному углу.
    for lon in range(int(BBOX[0] // 3 * 3), int(BBOX[2]) + 1, 3):
        for lat in range(int(BBOX[1] // 3 * 3), int(BBOX[3]) + 1, 3):
            name = f"ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif"
            url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}"
            ok = curl(url, CACHE / "worldcover" / name)
            log(f"  WorldCover {name}: {'есть' if ok else 'нет'}")


def aster_ged() -> None:
    # Адреса берутся из CMR, а не собираются по шаблону: гранулы переехали в
    # облако LP DAAC, а широта в имени — СЕВЕРНЫЙ край тайла, не южный.
    params = {"short_name": "AG100", "version": "003", "page_size": 100,
              "bounding_box": ",".join(map(str, BBOX))}
    url = "https://cmr.earthdata.nasa.gov/search/granules.json?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        entries = json.load(resp)["feed"]["entry"]
    log(f"  ASTER GED: гранул в CMR {len(entries)}")
    for e in entries:
        href = next((ln["href"] for ln in e["links"] if ln["href"].endswith(".h5")), None)
        if href is None:
            continue
        ok = curl(href, CACHE / "aster_ged" / href.rsplit("/", 1)[1], netrc=True)
        log(f"  ASTER GED {e['title']}: {'есть' if ok else 'нет'}")


def main() -> None:
    for name, job in (("глобальная геофизика GMT", world_geo), ("гидросеть OSM", osm_hydro),
                      ("ESA WorldCover", worldcover), ("ASTER GED", aster_ged)):
        log(f"--- {name} ---")
        try:
            job()
        except Exception as exc:  # noqa: BLE001 — один источник не должен ронять остальные
            log(f"ОШИБКА {name}: {exc!r}")
    log(f"Готово за {(time.time() - T0) / 60:.1f} мин.")


if __name__ == "__main__":
    main()
