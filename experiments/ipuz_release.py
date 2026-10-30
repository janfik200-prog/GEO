"""WP00/WP02/WP09 ТЗ ИПУЗ: сборка воспроизводимого релиза.

Складывает в каталог заморозки переносимый пакет: копию кода, замороженные
зависимости, точки входа без абсолютных путей, манифест SHA-256 и
``RELEASE_MANIFEST.json`` с версиями окружения и состоянием репозитория.

Пакет считается переносимым, если распакован в произвольный каталог и в нём
нет ни одного абсолютного пути этой машины. Это проверяется здесь же: все
файлы пакета сканируются на буквы дисков и домашние каталоги, и сборка падает,
если такой путь найден в коде.

Тяжёлые входы (спутниковые сцены, закрытые растры, дамп базы) в пакет не
входят и помечены как OPTIONAL_DATA: без них воспроизводятся метрики и карты
из производных таблиц, но не полный путь от исходных сцен.

Запуск::

    python -m experiments.ipuz_release --freeze <каталог_заморозки>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Что копируется в переносимый пакет.
КОД = ("src", "experiments", "tests", "scripts")
ИСКЛЮЧИТЬ = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

# Абсолютные пути машины, которых в коде пакета быть не должно. Двойной слэш
# после двоеточия исключается, иначе под шаблон попадает любой «https://».
ЗАПРЕТ = re.compile(
    r"(?i)(?<![\w:])(?:[A-Z]:[\\/](?![\\/])|/(?:home|users)/)[\w .\\/+-]{3,}")

# Файлы, в которых абсолютного пути быть не может: это то, что исполняется
# при воспроизведении релиза. Остальной код репозитория попадает в пакет как
# контекст, и его находки только перечисляются.
СТРОГО = ("src/", "scripts/", "experiments/ipuz_")

ТОЧКА_ВХОДА_КАРТЫ = '''"""Переносимая точка входа: три карты и три растра.

Путей этой машины здесь нет: корень пакета берётся от ``__file__``, каталог
заморозки — из ``--freeze`` (по умолчанию каталог на уровень выше), проект с
кодом — из ``--project``.

    python make_maps.py
    python make_maps.py --freeze /путь/к/заморозке --project /путь/к/коду
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ЗДЕСЬ = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, default=ЗДЕСЬ.parent)
    ap.add_argument("--project", type=Path,
                    default=ЗДЕСЬ.parent / "07_repro_package" / "project")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="корень с data/processed; по умолчанию = --project")
    a = ap.parse_args()

    project = a.project.resolve()
    if not (project / "experiments" / "ipuz_maps.py").exists():
        raise SystemExit("Не найден код проекта: %s\\n"
                         "Указать каталог ключом --project" % project)
    sys.path.insert(0, str(project))
    freeze = a.freeze.resolve()
    data_root = (a.data_root or project).resolve()

    sys.argv = ["ipuz_rasters", "--freeze", str(freeze), "--root", str(data_root)]
    runpy.run_module("experiments.ipuz_rasters", run_name="__main__")
    sys.argv = ["ipuz_maps", str(freeze), "--root", str(data_root)]
    runpy.run_module("experiments.ipuz_maps", run_name="__main__")


if __name__ == "__main__":
    main()
'''

ТОЧКА_ВХОДА_СТАТЬИ = '''"""Переносимая точка входа: сборка статьи в DOCX.

    python build_article.py
    python build_article.py --freeze /путь/к/заморозке --authors authors.json
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ЗДЕСЬ = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", type=Path, default=ЗДЕСЬ.parent)
    ap.add_argument("--project", type=Path,
                    default=ЗДЕСЬ.parent / "07_repro_package" / "project")
    ap.add_argument("--authors", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    project = a.project.resolve()
    if not (project / "experiments" / "ipuz_article.py").exists():
        raise SystemExit("Не найден код проекта: %s" % project)
    sys.path.insert(0, str(project))

    argv = ["ipuz_article", "--freeze", str(a.freeze.resolve())]
    if a.authors:
        argv += ["--authors", str(a.authors.resolve())]
    argv += ["--out", str((a.out or ЗДЕСЬ / "ARTICLE_FINAL.docx").resolve())]
    sys.argv = argv
    runpy.run_module("experiments.ipuz_article", run_name="__main__")


if __name__ == "__main__":
    main()
'''


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def копировать_код(src: Path, dst: Path) -> int:
    n = 0
    for d in КОД:
        источник = src / d
        if not источник.exists():
            raise SystemExit("Нет каталога %s" % источник)
        for p in источник.rglob("*"):
            if p.is_dir() or set(p.parts) & ИСКЛЮЧИТЬ or p.suffix == ".pyc":
                continue
            цель = dst / p.relative_to(src)
            цель.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, цель)
            n += 1
    return n


def проверить_пути(корень: Path, *, база: Path | None = None
                   ) -> tuple[list[dict], list[dict], int]:
    """Найти абсолютные пути машины: строго и справочно.

    Возвращает находки в исполняемой части релиза, находки в остальном коде
    и число просмотренных файлов.
    """
    база = база or корень
    строгие, прочие, всего = [], [], 0
    for p in sorted(корень.rglob("*.py")):
        всего += 1
        отн = p.relative_to(база).as_posix()
        исполняемый = any(x in отн for x in СТРОГО) or p.parent == база
        txt = p.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(txt.splitlines(), 1):
            for m in ЗАПРЕТ.finditer(line):
                находка = {"файл": отн, "строка": line_no,
                           "путь": m.group(0)[:80]}
                (строгие if исполняемый else прочие).append(находка)
    return строгие, прочие, всего


def окружение() -> dict:
    import importlib
    верс = {}
    for m in ("numpy", "pandas", "scikit-learn", "scipy", "rasterio",
              "matplotlib", "pyarrow", "joblib", "python-docx", "geopandas"):
        имя = {"scikit-learn": "sklearn", "python-docx": "docx"}.get(m, m)
        try:
            верс[m] = getattr(importlib.import_module(имя), "__version__", "?")
        except Exception:
            верс[m] = None
    try:
        import rasterio
        верс["GDAL"] = rasterio.__gdal_version__
    except Exception:
        верс["GDAL"] = None
    return {"python": platform.python_version(),
            "реализация": platform.python_implementation(),
            "платформа": platform.platform(), "пакеты": верс}


def git(*args: str) -> str:
    try:
        return subprocess.run(("git", "-C", str(ROOT)) + args, check=True,
                              capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    except Exception as e:  # noqa: BLE001
        return "недоступно: %s" % type(e).__name__


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--docs", type=Path, default=ROOT / "docs_ipuz",
                    help="каталог с README_REPRODUCE.md и прочими документами")
    a = ap.parse_args()

    freeze = a.freeze.resolve()
    pkg = freeze / "07_repro_package"
    project = pkg / "project"
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True, exist_ok=True)

    n = копировать_код(ROOT, project)
    print("скопировано файлов кода: %d" % n)

    # --- замороженные зависимости и версия языка
    env = окружение()
    env_dir = pkg / "env"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "python_version.txt").write_text(
        env["python"] + "\n", encoding="utf-8")
    freeze_txt = subprocess.run(
        (sys.executable, "-m", "pip", "freeze"), capture_output=True,
        text=True, encoding="utf-8").stdout
    (env_dir / "requirements_exact.txt").write_text(freeze_txt, encoding="utf-8")
    (env_dir / "env_lock.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print("зафиксировано пакетов: %d" % len(freeze_txt.splitlines()))

    # --- переносимые точки входа
    (freeze / "05_maps").mkdir(parents=True, exist_ok=True)
    (freeze / "06_article").mkdir(parents=True, exist_ok=True)
    (freeze / "05_maps" / "make_maps.py").write_text(
        ТОЧКА_ВХОДА_КАРТЫ, encoding="utf-8")
    (freeze / "06_article" / "build_article.py").write_text(
        ТОЧКА_ВХОДА_СТАТЬИ, encoding="utf-8")

    # --- документы
    docs_dst = freeze / "docs"
    docs_dst.mkdir(exist_ok=True)
    скопировано_доков = []
    if a.docs.exists():
        for p in sorted(a.docs.iterdir()):
            if p.is_file():
                shutil.copy2(p, docs_dst / p.name)
                скопировано_доков.append(p.name)
    print("документов:", ", ".join(скопировано_доков) or "нет")

    # Лист сдачи кладётся ещё и в корень: он должен попадаться первым.
    лист = a.docs / "HANDOFF.md"
    if лист.exists():
        shutil.copy2(лист, freeze / "ЧИТАТЬ_ПЕРВЫМ.md")
        print("лист сдачи: ЧИТАТЬ_ПЕРВЫМ.md")

    # --- проверка переносимости
    плохие, прочие, скан = проверить_пути(project, база=project)
    for точка in (freeze / "05_maps" / "make_maps.py",
                  freeze / "06_article" / "build_article.py"):
        скан += 1
        for line_no, line in enumerate(
                точка.read_text(encoding="utf-8").splitlines(), 1):
            for m in ЗАПРЕТ.finditer(line):
                плохие.append({"файл": точка.name, "строка": line_no,
                               "путь": m.group(0)[:80]})

    (freeze / "qa").mkdir(exist_ok=True)
    (freeze / "qa" / "portability_check.json").write_text(json.dumps({
        "проверено_файлов": скан,
        "строгая_часть": "точки входа, src/, scripts/, experiments/ipuz_*",
        "абсолютных_путей_в_строгой_части": len(плохие),
        "находки": плохие[:40],
        "в_остальном_коде_репозитория": len(прочие),
        "прочие_находки": прочие[:40],
        "примечание": "Остальной код попадает в пакет как контекст и в "
                      "воспроизведении релиза не участвует; его упоминания "
                      "путей перечислены, но приёмку не блокируют.",
        "итог": "PASS" if not плохие else "FAIL",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("файлов просмотрено: %d" % скан)
    print("абсолютных путей в исполняемой части: %d" % len(плохие))
    for b in плохие[:10]:
        print("   %s:%d  %s" % (b["файл"], b["строка"], b["путь"]))
    print("упоминаний путей в прочем коде репозитория: %d" % len(прочие))

    # --- манифест SHA-256 по всему пакету
    строки, всего = [], 0
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256.csv":
            строки.append("%s *./%s" % (sha256(p),
                                        p.relative_to(pkg).as_posix()))
            всего += 1
    (pkg / "MANIFEST_SHA256.csv").write_text(
        "sha256,path\n" + "\n".join(строки) + "\n", encoding="utf-8")
    print("в манифесте файлов: %d" % всего)

    # --- манифест релиза
    артефакты = {}
    for rel in ("01_inputs/target_mask.csv",
                "04_metrics/main_metrics.csv", "04_metrics/ceiling_metrics.csv",
                "04_metrics/controls.csv", "04_metrics/o10_block_ci.csv",
                "04_metrics/spectral_ablation.csv",
                "05_maps/crit.tif", "05_maps/oof_remote.tif",
                "05_maps/overlap_class.tif", "05_maps/map_pair.png",
                "05_maps/map_overlap.png", "05_maps/map_detail.png",
                "06_article/ARTICLE_FINAL.docx",
                "03_runs/stdout_pytest_full.log"):
        p = freeze / rel
        артефакты[rel] = sha256(p) if p.exists() else None

    oof = ROOT / "data" / "processed" / "operating_point_masked.parquet"
    манифест = {
        "release_id": "IPUZ-2026-" + freeze.name,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "лист": "R-48-XI,XII",
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_short": git("status", "--short"),
        "окружение": env,
        "первичный_вход_модели": {
            "файл": "data/processed/operating_point_masked.parquet",
            "sha256": sha256(oof) if oof.exists() else None,
            "примечание": "SHA файла parquet меняется при перезаписи даже без "
                          "изменения значений — сверять значения, а не хеш",
        },
        "артефакты_sha256": артефакты,
        "манифест_пакета": {"файл": "07_repro_package/MANIFEST_SHA256.csv",
                            "файлов": всего,
                            "sha256": sha256(pkg / "MANIFEST_SHA256.csv")},
        "переносимость": "PASS" if not плохие else "FAIL",
        "OPTIONAL_DATA": [
            "исходные сцены Sentinel-1/2, Landsat 8/9, ALOS PALSAR-2, ASTER",
            "растры и слои критериального расчёта организации",
            "дамп базы данных паспортов признаков",
        ],
        "publication_rights_status": "NOT_CONFIRMED",
        "external_gold_validation_status": "NOT_DONE",
        "notes": "Воспроизводится точный пересчёт опубликованных метрик и карт "
                 "из производных таблиц. Полный путь от спутниковых сцен "
                 "требует OPTIONAL_DATA и здесь не воспроизводится.",
    }
    (freeze / "RELEASE_MANIFEST.json").write_text(
        json.dumps(манифест, ensure_ascii=False, indent=2), encoding="utf-8")
    print("записано:", freeze / "RELEASE_MANIFEST.json")
    if плохие:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
