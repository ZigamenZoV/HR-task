"""Сопоставление строки с эталоном: улица -> дома -> признаки -> скор -> отказ."""
import json
import math

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from .index import EtalonIndex
from .query import QueryParser
from .text import key, stem_key, full_key
from .numbers import split_house

TOP_STREETS = 25
MAX_SPAN = 6
# Ничья: разные адреса с одинаковым скором (см. _n_tied). Порог почти нулевой намеренно.
# При 0.1 он задевал «солнечная 8»: ул. Солнечная и б-р Солнечный отличаются на 0.07,
# и это настоящее различие, а не ничья. В разметке минимальный отрыв верного top-1 1.0.
TIE_EPS = 1e-3

# Порядок признаков кандидата. Скор кандидата: линейная комбинация с весами из model.json.
CAND_FEATURES = [
    "street_ratio",      # лучший Левенштейн-ratio подотрезка фразы и варианта улицы (стемы)
    "street_partial",    # partial_ratio по полному ключу: ловит склейки
    "partial_short",     # partial для коротких названий (< 6 символов) отдельно: там он шумный
    "coverage",          # доля символов улицы во фразе, объяснённая лучшим подотрезком
    "gap_to_best",       # отставание улицы от лучшей улицы запроса
    "type_match", "type_conflict",
    "house_exact", "house_base", "house_suffix_sim", "no_house",
    "city_match", "city_conflict",
    "mention_match", "mention_conflict",
]
# Признаки уровня запроса для решения об отказе. city_other («город из поля не из
# эталона») считается, но в модель не входит: на CV прироста ноль, а при сдвиге
# (у позитива кривое поле города) он режет верные ответы, см. scripts/stress.py.
REJECT_FEATURES_ALL = [
    "top_score", "margin", "street_ratio", "coverage", "house_exact", "house_base",
    "no_house", "city_other", "city_empty", "n_words", "unexplained_words",
]
REJECT_FEATURES = [f for f in REJECT_FEATURES_ALL if f != "city_other"]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class Matcher:
    def __init__(self, index: EtalonIndex, model: dict = None):
        self.idx = index
        self.parser = QueryParser(index.street_word_keys, index.cities)
        self.model = model or default_model()
        for k, names in (("cand_features", CAND_FEATURES), ("reject_features", REJECT_FEATURES)):
            if k in self.model and self.model[k] != names:
                raise ValueError(f"model.json устарел: {k} не совпадает с кодом, перезапустите train.py")
        self._alias_sid = np.array(index.alias_sid)
        self._n_streets = len(index.streets)
        # короткое название (< 6 символов ключа): partial_ratio на нём шумит
        first_alias = {}
        for i, sid in enumerate(index.alias_sid):
            first_alias.setdefault(sid, i)
        self._short = np.array([len(index.alias_full[first_alias[s]]) < 6
                                for s in range(self._n_streets)])

    @classmethod
    def load(cls, etalon_path, model_path=None):
        model = None
        if model_path:
            with open(model_path, encoding="utf-8") as f:
                model = json.load(f)
        return cls(EtalonIndex.from_jsonl(etalon_path), model)

    # ------------------------------------------------------------------ город
    def city_field(self, city):
        """-> (канонический город | None, статус 'match'|'empty'|'other')."""
        c = (city or "").strip()
        if not c:
            return None, "empty"
        k = key(c)
        best = process.extractOne(k, list(self.idx.city_keys), scorer=Levenshtein.distance)
        if best and best[1] <= (2 if len(k) >= 6 else 1):
            return self.idx.city_keys[best[0]], "match"
        return None, "other"

    # ------------------------------------------------------------------ улицы
    def street_scores(self, toks):
        """Для каждой улицы: ratio, partial, coverage. Векторно через cdist."""
        n = self._n_streets
        ratio = np.zeros(n); partial = np.zeros(n); cover = np.zeros(n)
        if not toks:
            return ratio, partial, cover
        spans, span_len = [], []
        total = len(stem_key(toks)) or 1
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + MAX_SPAN) + 1):
                sk = stem_key(toks[i:j])
                spans.append(sk); span_len.append(len(sk))
        m = process.cdist(spans, self.idx.alias_stem, scorer=fuzz.ratio, workers=1)
        best_span = m.argmax(axis=0)
        a_ratio = m.max(axis=0) / 100.0
        a_cover = np.array(span_len)[best_span] / total
        a_partial = process.cdist([full_key(toks)], self.idx.alias_full,
                                  scorer=fuzz.partial_ratio, workers=1)[0] / 100.0
        # максимум по вариантам одной улицы
        np.maximum.at(ratio, self._alias_sid, a_ratio)
        np.maximum.at(partial, self._alias_sid, a_partial)
        # покрытие берём у варианта с лучшим ratio
        order = np.lexsort((a_cover, a_ratio))
        cover[self._alias_sid[order]] = a_cover[order]
        return ratio, partial, cover

    # ------------------------------------------------------------------ дом
    @staticmethod
    def house_feats(q, canon, base, suf):
        if not q.house:
            return 0.0, 0.0, 0.0
        best = (0.0, 0.0, 0.0)
        for h in (q.house, *q.house_alts):
            if h == canon:
                return 1.0, 1.0, 1.0
            qb, qs = split_house(h)
            if qb is not None and qb == base:
                sim = 1.0 - Levenshtein.distance(qs, suf) / max(len(qs), len(suf), 1)
                best = max(best, (0.0, 1.0, sim))
        return best

    # ------------------------------------------------------------------ основной вызов
    def candidates(self, raw, city, channel=None):
        q = self.parser.parse(raw)
        city_c, city_status = self.city_field(city)
        ratio, partial, cover = self.street_scores(q.street_tokens)
        combined = np.maximum(ratio, partial * 0.9)
        top = np.argsort(-combined)[:TOP_STREETS]
        best_ratio = ratio.max() if len(ratio) else 0.0
        rows, meta = [], []
        for sid in top:
            s = self.idx.streets[sid]
            short = bool(self._short[sid])
            for eid, canon, base, suf in s.houses:
                hx, hb, hs = self.house_feats(q, canon, base, suf)
                f = {
                    "street_ratio": ratio[sid],
                    "street_partial": 0.0 if short else partial[sid],
                    "partial_short": partial[sid] if short else 0.0,
                    "coverage": cover[sid],
                    "gap_to_best": ratio[sid] - best_ratio,
                    "type_match": float(bool(q.street_type) and q.street_type == s.stype),
                    "type_conflict": float(bool(q.street_type) and q.street_type != s.stype),
                    "house_exact": hx, "house_base": hb, "house_suffix_sim": hs,
                    "no_house": float(not q.house),
                    "city_match": float(city_c == s.city),
                    "city_conflict": float(city_c is not None and city_c != s.city),
                    "mention_match": float(q.city_mention == s.city),
                    "mention_conflict": float(bool(q.city_mention) and q.city_mention != s.city),
                }
                rows.append([f[n] for n in CAND_FEATURES])
                meta.append(eid)
        X = np.array(rows) if rows else np.zeros((0, len(CAND_FEATURES)))
        ctx = {"query": q, "city_status": city_status, "channel": channel}
        return meta, X, ctx

    def score(self, X):
        w = np.array(self.model["cand_w"]); b = self.model["cand_b"]
        return X @ w + b if len(X) else np.zeros(0)

    def reject_features(self, meta, X, s, order, ctx, names=None):
        q = ctx["query"]
        if len(order) == 0:
            top = dict.fromkeys(CAND_FEATURES, 0.0); top_score = -10.0; margin = 0.0
        else:
            top = dict(zip(CAND_FEATURES, X[order[0]]))
            top_score = s[order[0]]
            # отрыв от лучшего конкурента, не считая дублей эталона (у них скор одинаковый)
            grp = self.idx.dup_of.get(meta[order[0]], meta[order[0]])
            rivals = [s[i] for i in order[1:] if self.idx.dup_of.get(meta[i], meta[i]) != grp]
            margin = top_score - rivals[0] if rivals else 5.0
        explained = top["coverage"] * q.n_words
        f = {
            "top_score": top_score, "margin": min(margin, 5.0),
            "street_ratio": top["street_ratio"], "coverage": top["coverage"],
            "house_exact": top["house_exact"], "house_base": top["house_base"],
            "no_house": float(not q.house),
            "city_other": float(ctx["city_status"] == "other"),
            "city_empty": float(ctx["city_status"] == "empty"),
            "n_words": min(q.n_words, 6) / 6.0,
            "unexplained_words": min(max(q.n_words - explained, 0.0), 4) / 4.0,
        }
        return [f[n] for n in (names or REJECT_FEATURES)]

    def match(self, raw, city, channel=None, k=3):
        """-> (список etalon_id, список скоров). Пустой список = отказ."""
        meta, X, ctx = self.candidates(raw, city, channel)
        s = self.score(X)
        order = self._rank(meta, s)
        rf = np.array(self.reject_features(meta, X, s, order, ctx))
        p_ok = _sigmoid(float(rf @ np.array(self.model["rej_w"]) + self.model["rej_b"]))
        p_ok = min(p_ok, 1.0 / self._n_tied(meta, s, order))
        if p_ok < self.model["threshold"] or len(order) == 0:
            return [], []
        top = order[:k]
        # скоры в [0,1]: у первого P(top-1 верен), у остальных она же, умноженная на exp(s_i - s_1)
        ex = np.exp(s[top] - s[top[0]])
        return [meta[i] for i in top], [round(p_ok * float(e), 4) for e in ex]

    def _n_tied(self, meta, s, order):
        """Сколько разных адресов делят первое место (дубли эталона считаются одним).

        Ничья возникает, когда одна и та же улица с тем же домом есть в двух городах,
        а города в запросе нет: «Лескова 21» (Новосибирск и Нижний Новгород).
        Модель отказа ничьих почти не видит: вес отрыва у неё маленький, потому что
        в разметке ничьих нет. Но при k равных кандидатах честная вероятность
        верного top-1 не выше 1/k, поэтому P ограничивается сверху.
        """
        if len(order) == 0:
            return 1
        top = s[order[0]]
        groups = {self.idx.dup_of.get(meta[i], meta[i]) for i in order if top - s[i] < TIE_EPS}
        return max(len(groups), 1)

    def _rank(self, meta, s):
        """Сортировка по скору; дубли эталона кладём рядом, чтобы оба попали в top-3."""
        order = list(np.argsort(-s, kind="stable"))
        seen, out = set(), []
        for i in order:
            eid = meta[i]
            if eid in seen:
                continue
            out.append(i); seen.add(eid)
            grp = self.idx.dup_of.get(eid)
            if grp:
                for j in order:
                    if meta[j] not in seen and self.idx.dup_of.get(meta[j]) == grp:
                        out.append(j); seen.add(meta[j])
        return out


def default_model():
    """Ручные веса: стартовая точка до обучения."""
    w = dict.fromkeys(CAND_FEATURES, 0.0)
    w.update(street_ratio=4.0, street_partial=1.0, partial_short=0.3, coverage=0.5,
             gap_to_best=2.0, type_match=0.3, type_conflict=-0.2,
             house_exact=2.5, house_base=1.5, house_suffix_sim=0.3,
             city_match=0.6, city_conflict=-0.3, mention_match=0.8, mention_conflict=-0.8)
    r = dict.fromkeys(REJECT_FEATURES, 0.0)
    r.update(top_score=1.0)
    return {"cand_w": [w[n] for n in CAND_FEATURES], "cand_b": 0.0,
            "rej_w": [r[n] for n in REJECT_FEATURES], "rej_b": -5.0, "threshold": 0.5}
