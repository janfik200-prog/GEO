"""Скачивание смежных листов Госгеолкарты-200 (Анабарская серия) с geokniga.

Зачем. Узкое место проекта — заверка: на нашем листе R-48-XI,XII всего 4 точки
«Проявления» (все урановые) и 19 строго несмещённых ячеек, из-за чего квант
метрики lift@10% равен 0.53. Точки с СОСЕДНИХ листов — единственный доступный
способ получить независимые метки, не трогая стоп-лист признаков: они лежат в тех
же структурах Анабарского щита, но вне обучающей площади.

Что качается. Для каждого смежного листа — весь комплект с geokniga: карта (JPG),
GeoTIFF привязанной карты, условные обозначения, стратколонка, разрез, а также
карта полезных ископаемых там, где она выложена. Именно карта полезных ископаемых
несёт точки проявлений; на части листов серии её нет, и это фиксируется в отчёте,
а не замалчивается.

Соседство считается по номенклатуре: R-48 делится на 6x6 листов 1:200 000,
XI,XII — второй ряд, столбцы 5-6, поэтому север — V,VI, юг — XVII,XVIII,
запад — IX,X, а восток уходит в соседний миллионный лист R-49 (VII,VIII).

Идемпотентно: уже скачанные файлы не перекачиваются, битые (HTML-заглушки) —
удаляются и качаются заново.

Результат прогона 05.08.2026 (249 МБ, 7 смежных листов из 8 возможных; северо-
западный R-48-III,IV в каталоге geokniga отсутствует):

- геологические карты пришли георастрами EPSG:4326, охваты стыкуются точно
  (наш лист 106-108 в.д., 70.67-71.33 с.ш.);
- отдельная «Карта полезных ископаемых» выложена только на R-48-XV,XVI, но она и
  не нужна: раздел «Проявления полезных ископаемых» напечатан прямо в легенде
  геологической карты, и точки снимаются с георастра;
- ГЛАВНОЕ по существу заверки: золото как рудный объект есть только на восточных
  листах R-49 (Оленёкское поднятие) — на R-49-I,II нанесены «минерализованные
  зоны (золото)», на R-49-VII,VIII и R-49-XIII,XIV золото идёт в шлихах. На
  собственном листе R-48-XI,XII в легенде золота нет вообще: проявления —
  Ta,Nb и TR, аномалии — Pb, Zn, Zr.

Запуск из корня: ``python -m experiments.fetch_ggk200``.
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_paths import GGK200  # noqa: E402

BASE = "https://www.geokniga.org"

# лист -> (id страницы geokniga, положение относительно R-48-XI,XII)
SHEETS = {
    "R-48-V,VI":       (43904, "север"),
    "R-48-IX,X":       (43903, "запад"),
    "R-48-XV,XVI":     (785,   "юго-запад"),
    "R-48-XVII,XVIII": (43908, "юг"),
    "R-49-I,II":       (43919, "северо-восток"),
    "R-49-VII,VIII":   (43921, "восток"),
    "R-49-XIII,XIV":   (43922, "юго-восток"),
    "R-48-XI,XII":     (784,   "САМ ЛИСТ"),
}

# Ссылки на файлы geokniga отдаёт в двух видах: /sites/geokniga/files/maps/<slug>
# для основной карты и /sites/geokniga/../../maps/additional/<slug> для врезок.
# Второй вид обязан нормализоваться в /maps/additional/<slug>: если срезать
# ``../../`` неверно, сервер молча отдаёт HTML-страницу «файл не найден» с кодом
# 200, и на диск ложится мусор под именем .jpg — поэтому тип содержимого ниже
# проверяется явно.
LINK_RE = re.compile(
    r'href="(https://www\.geokniga\.org/sites/geokniga/[^"]+)"[^>]*>(.{0,150}?)</a>',
    re.S)
BINARY_TYPES = ("image/", "application/octet-stream", "application/pdf")


def page_files(node_id: int) -> list[tuple[str, str]]:
    """Список (подпись, URL) файлов на странице листа; превью отбрасываются."""
    req = urllib.request.Request(f"{BASE}/maps/{node_id}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        page = resp.read().decode("utf-8", "replace")
    out, seen = [], set()
    for url, raw in LINK_RE.findall(page):
        if "imagecache" in url:                      # эскиз карточки, не файл
            continue
        url = url.replace("/sites/geokniga/../../", "/")
        if url in seen:
            continue
        seen.add(url)
        out.append((re.sub(r"<[^>]+>", "", html.unescape(raw)).strip(), url))
    return out


def is_binary(path: Path) -> bool:
    """Файл — картинка/растр, а не HTML-заглушка (по сигнатуре первых байтов)."""
    with path.open("rb") as f:
        head = f.read(8)
    return head[:3] == b"\xff\xd8\xff" or head[:4] in (b"II*\x00", b"MM\x00*",
                                                       b"%PDF")


def download(url: str, dest: Path) -> bool:
    """Скачать файл, если его ещё нет; при обрыве — докачать.

    Заглушка (HTML под именем .jpg) удаляется, а не остаётся в кэше: иначе
    повторный запуск примет её за успешную загрузку и ошибка законсервируется.
    """
    if dest.exists() and dest.stat().st_size > 10_000 and is_binary(dest):
        return True
    dest.unlink(missing_ok=True)
    subprocess.run(["curl", "-sS", "-L", "--connect-timeout", "20",
                    "--max-time", "1800", "--retry", "5", "--retry-all-errors",
                    "-o", str(dest), url],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not (dest.exists() and dest.stat().st_size > 10_000 and is_binary(dest)):
        dest.unlink(missing_ok=True)
        return False
    return True


def main() -> None:
    print(f"Кэш: {GGK200}\n")
    report = []
    for sheet, (node, side) in SHEETS.items():
        folder = GGK200 / sheet.replace(",", "_")
        folder.mkdir(parents=True, exist_ok=True)
        files = page_files(node)
        print(f"=== {sheet} ({side}), geokniga/maps/{node}: файлов {len(files)}")
        ok, mb, has_pi = 0, 0.0, False
        for label, url in files:
            dest = folder / url.rsplit("/", 1)[-1]
            if download(url, dest):
                ok += 1
                mb += dest.stat().st_size / 1e6
                has_pi |= "полезных ископаемых" in label.lower()
                print(f"    {label[:38]:38s} {dest.name[:52]:52s} "
                      f"{dest.stat().st_size / 1e6:6.1f} МБ")
            else:
                print(f"    {label[:38]:38s} ОШИБКА: {url}")
        report.append((sheet, side, ok, len(files), mb, has_pi))
        print()

    print("=== Итог ===")
    print(f"{'лист':<17} {'сторона':<14} {'файлов':>7} {'МБ':>7}  карта ПИ")
    for sheet, side, ok, total, mb, has_pi in report:
        print(f"{sheet:<17} {side:<14} {ok:>3}/{total:<3} {mb:>7.1f}  "
              f"{'есть' if has_pi else '—'}")
    n_pi = sum(1 for r in report if r[5])
    print(f"\nКарта полезных ископаемых выложена на {n_pi} листах из {len(report)}; "
          "точки проявлений снимаются только с неё.")


if __name__ == "__main__":
    main()
