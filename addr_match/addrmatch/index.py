"""Индекс эталона: уникальные улицы, их варианты написания и дома.

Эталон строится один раз при старте и в замер задержки не входит.
"""
import json
from dataclasses import dataclass, field

from .text import key, stem_key, full_key, tokenize, merge_ordinals, TYPE_BY_KEY
from .numbers import split_house

# Звания и «имени»: люди их часто опускают, эталон пишет полностью.
_TITLES = {key(w) for w in """им имени маршала генерала героя героев советского союза
академика космонавта гвардии сержанта летописца""".split()}
_NOISE = {key(w) for w in "им имени".split()}


@dataclass
class Street:
    sid: int
    city: str
    stype: str
    name: str
    aliases: list = field(default_factory=list)       # списки токенов
    houses: list = field(default_factory=list)        # (etalon_id, canon, base, suffix)


def _name_tokens(name):
    toks = []
    for t in merge_ordinals(tokenize(name)):
        k = key(t)
        if len(t) == 1 and not t.isdigit():
            continue                                  # инициалы: «В.И.», «Тимме Я»
        if k in TYPE_BY_KEY or k in _NOISE:
            continue                                  # «Невельского пр-д», «им.газеты»
        toks.append(t)
    return toks


def _aliases(name):
    base = _name_tokens(name)
    out = [base]
    stripped = [t for t in base if key(t) not in _TITLES]
    if stripped and stripped != base and sum(map(len, stripped)) >= 5:
        out.append(stripped)                          # «Героя Советского Союза Прыгунова» -> «Прыгунова»
    raw = [t for t in merge_ordinals(tokenize(name)) if not (len(t) == 1 and not t.isdigit())]
    if raw != base and raw:
        out.append(raw)                               # «им газеты ...» тоже бывает сказано
    return out


class EtalonIndex:
    def __init__(self, rows):
        self.rows = rows
        self.by_id = {r["etalon_id"]: r for r in rows}
        streets = {}
        for r in rows:
            sk = (r["city"], r["street_type"], r["street"])
            if sk not in streets:
                streets[sk] = Street(len(streets), *sk, aliases=_aliases(r["street"]))
            canon = r["house"].lower().replace(" ", "")
            base, suf = split_house(canon)
            streets[sk].houses.append((r["etalon_id"], canon, base, suf))
        self.streets = list(streets.values())
        # плоские массивы вариантов для rapidfuzz.cdist
        self.alias_sid, self.alias_stem, self.alias_full = [], [], []
        for s in self.streets:
            for a in s.aliases:
                self.alias_sid.append(s.sid)
                self.alias_stem.append(stem_key(a))
                self.alias_full.append(full_key(a))
        self.street_word_keys = {key(t) for s in self.streets for a in s.aliases
                                 for t in a if len(t) >= 3 and not t.isdigit()}
        self.cities = sorted({r["city"] for r in rows})
        self.city_keys = {key(c): c for c in self.cities}
        # дубли адресов в эталоне: одинаковые город+улица+дом под разными id
        seen = {}
        self.dup_of = {}
        for r in rows:
            k = (r["city"], r["street_type"], r["street"], r["house"].lower())
            if k in seen:
                self.dup_of[r["etalon_id"]] = seen[k]
                self.dup_of.setdefault(seen[k], seen[k])
            else:
                seen[k] = r["etalon_id"]

    @classmethod
    def from_jsonl(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])
