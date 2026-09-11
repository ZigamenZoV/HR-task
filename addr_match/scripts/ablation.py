"""Абляции: что даёт каждый шаг. Запуск: python scripts/ablation.py

Каждый вариант запускается отдельным процессом с флагом окружения, потому что словари
ключей строятся при импорте. top1/top3 считаются без отказа (чистое ранжирование), with_reject с моделью отказа,
обученной на полном пайплайне (для абляций она не переобучалась, это оценка снизу).
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = r'''
import json, sys
sys.path.insert(0, ROOT)
from addrmatch.matcher import Matcher
from addrmatch import metrics
M = Matcher.load(ROOT + "/data/etalon.jsonl", ROOT + "/models/model.json")
rows = [json.loads(l) for l in open(ROOT + "/data/adresses_labeled.jsonl", encoding="utf-8")]
full = metrics.compute(rows, {r["id"]: M.match(r["raw_adress"], r["city"], r["channel"])[0] for r in rows})
M.model["threshold"] = 0.0
preds = {r["id"]: M.match(r["raw_adress"], r["city"], r["channel"])[0] for r in rows}
by = {}
for ch in ("voice", "webchat"):
    sub = [r for r in rows if r["channel"] == ch]
    by[ch] = metrics.compute(sub, preds)["top1"]
m = metrics.compute(rows, preds)
print(json.dumps({"top1": m["top1"], "top3": m["top3"], "top1_voice": by["voice"],
                  "top1_webchat": by["webchat"],
                  "with_reject": {k: full[k] for k in ("top1", "reject_precision", "reject_recall")}}))
'''.replace("ROOT", repr(ROOT))

VARIANTS = [
    ("полный пайплайн", {}),
    ("без фонетического ключа", {"ADDR_NO_PHONETIC": "1"}),
    ("без стемминга окончаний", {"ADDR_NO_STEM": "1"}),
    ("без починки склеек/разрывов", {"ADDR_NO_GLUE": "1"}),
    ("без ключа, стемов и склеек", {"ADDR_NO_PHONETIC": "1", "ADDR_NO_STEM": "1", "ADDR_NO_GLUE": "1"}),
]

for name, env in VARIANTS:
    out = subprocess.run([sys.executable, "-c", CODE], env={**os.environ, **env},
                         capture_output=True, text=True, encoding="utf-8")
    res = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else out.stderr[-300:]
    print(f"{name:32s} {res}")
