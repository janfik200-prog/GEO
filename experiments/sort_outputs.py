"""Раскладка картинок из `outputs/` по папкам задач.

Правила разбора живут в `src/outputs.py`. Скрипт переносит всё, что лежит в
корне `outputs/`, в папку своей задачи и печатает файлы, не попавшие ни под
одно правило, — их надо либо переименовать, либо дописать правило.

Служебные папки (`logs`, `metrics`) не трогаются: там имена файлов и так
привязаны к скриптам, а раскладка нужна именно для картинок.

Запуск из корня: ``python -m experiments.sort_outputs`` (сухой прогон:
``python -m experiments.sort_outputs --показать``).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KEEP = {"logs", "metrics"}
EXT = {".png", ".jpg", ".svg", ".pdf", ".zip", ".html"}


def main() -> None:
    dry = "--показать" in sys.argv
    moved, unknown, clash = 0, [], []
    for src in sorted(outputs.OUT.iterdir()):
        if src.is_dir() or src.suffix.lower() not in EXT:
            continue
        task = outputs.task_of(src.name)
        if task == outputs.OTHER:
            unknown.append(src.name)
        dst = outputs.OUT / task / src.name
        if dst.exists():
            clash.append(src.name)
            continue
        print("%-46s -> %s" % (src.name, task))
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        moved += 1

    print()
    print("перенесено файлов: %d%s" % (moved, " (сухой прогон)" if dry else ""))
    if unknown:
        print("без правила (легли в «%s»): %s" % (outputs.OTHER,
                                                  ", ".join(unknown)))
    if clash:
        print("уже есть в папке задачи, не трогал: %s" % ", ".join(clash))
    for folder in sorted(p for p in outputs.OUT.iterdir()
                         if p.is_dir() and p.name not in KEEP):
        print("  %-30s %d файлов" % (folder.name,
                                     len(list(folder.glob("*")))))


if __name__ == "__main__":
    main()
