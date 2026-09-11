"""Разбор ответа кандидата: улица, дом, тип улицы, упомянутый город.

Порядок:
1. токенизация (пунктуация, «ленина54» -> «ленина 54», «2-я» -> «2я»);
2. отрезается хвост про квартиру/подъезд после номера («28 кв 12»);
   затем починка склеек и разрывов через словарь ключей
   («шоссемосковское», «еслизнаете», «пятьдесятдва», «одинна дцать»);
3. номер дома ищется с конца фразы: числа, буквы корпуса, «дробь», «дом»,
   паразиты внутри группы пропускаются. Всё, что левее, считается улицей,
   поэтому числа в названиях («9 мая», «60 лет ...») не путаются с домом;
4. из улицы убираются паразиты, тип улицы и название города.
"""
import os
import re
from dataclasses import dataclass, field

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .text import key, tokenize, merge_ordinals, FILLERS, TYPE_BY_KEY, LONG_TYPE_KEYS, ETALON_CITIES
from .numbers import (word_to_num, NUM_BY_KEY, LETTERS, is_korpus, DOM_RAW,
                      render_house, compose, _HOUSE_TOKEN)

_NO_GLUE = bool(os.environ.get("ADDR_NO_GLUE"))
# Хвост после номера дома, который к адресу дома не относится: «28 кв 12»,
# «двадцать восемь квартира двенадцать», «подъезд 3». В размеченных данных таких
# строк нет, это страховка для живого трафика. Сравнение по сырому тексту.
_TAIL_RAW = {"кв", "квартира", "квартиры", "подъезд", "подьезд", "этаж", "офис", "оф",
             "помещение", "пом"}
_ORDINAL = re.compile(r"^\d+(я|й|го|ая|ой|ий|ый|е|ое|ю)$")
# Разговорные формы городов. Берутся только те, чей город есть в эталоне.
# Сравниваются по сырому тексту: по фонетическому ключу «нижний» совпадает с «нижне»
# из «Нижне-Печерской» и «Нижне-Волжской», и название улицы теряло половину.
_CITY_SLANG = {"нижний": "Нижний Новгород", "нижнем": "Нижний Новгород", "екб": "Екатеринбург"}


@dataclass
class Query:
    raw: str
    street_tokens: list = field(default_factory=list)
    house: str = ""                 # каноника «381к3», «» если дома нет
    house_alts: tuple = ()          # альтернативные прочтения («ка» = буква К или корпус А)
    street_type: str = ""           # канонический тип, если назван
    city_mention: str = ""          # город эталона, названный в самой строке
    n_words: int = 0                # слов после чистки (для признаков отказа)


class QueryParser:
    def __init__(self, street_word_keys, cities=ETALON_CITIES):
        self.city_keys = {key(c): c for c in cities}
        self.city_slang = {w: c for w, c in _CITY_SLANG.items() if c in cities}
        self.street_keys = set(street_word_keys)
        self.street_keys_list = sorted(self.street_keys)
        self.vocab = (self.street_keys | FILLERS | set(TYPE_BY_KEY) | set(NUM_BY_KEY)
                      | set(LONG_TYPE_KEYS) | {key("дом")})

    # ------------------------------------------------------------ классы токенов
    @staticmethod
    def _type_of(tok):
        k = key(tok)
        if k in TYPE_BY_KEY:
            return TYPE_BY_KEY[k]
        if len(k) >= 5:
            for tk, t in LONG_TYPE_KEYS.items():
                if Levenshtein.distance(k, tk) <= 1:
                    return t
        return None

    def _near_street(self, k):
        if k in self.street_keys:
            return True
        hit = process.extractOne(k, self.street_keys_list, scorer=Levenshtein.distance,
                                 score_cutoff=1)
        return hit is not None

    # ------------------------------------------------------------ починка склеек
    def _known(self, k):
        return k in self.vocab

    def _strong(self, part, side):
        k = key(part)
        if k in NUM_BY_KEY and (len(k) >= 4 or (side == "right" and part == "сто")):
            return True
        if k in TYPE_BY_KEY and len(k) >= 4:
            return True
        return k in self.vocab and len(k) >= 4

    def _split(self, tok, depth=2):
        """Одна склейка -> список частей. Режем только слова, которых нет в словаре."""
        if len(tok) >= 2 and tok[0] == "к" and tok != "ка" and (
                (tok[1:] in LETTERS and len(tok) == 2)
                or word_to_num(tok[1:], fuzzy=False) is not None):
            return ["к", tok[1:]]                 # «кдва», «кб» -> корпус + ...
        k = key(tok)
        if (depth == 0 or len(tok) < 5 or self._known(k) or word_to_num(tok) is not None
                or self._type_of(tok) or self._near_street(k) or tok.isdigit()):
            return [tok]
        best, best_score = None, 0
        for i in range(1, len(tok)):
            left, right = tok[:i], tok[i:]
            kl, kr = key(left), key(right)
            both = self._known(kl) and self._known(kr)
            ok = both or (self._strong(left, "left") and len(right) >= 3) or (
                self._strong(right, "right") and len(left) >= 3)
            if not ok:
                continue
            score = (len(left) if self._known(kl) else 0) + (len(right) if self._known(kr) else 0)
            score += 100 if both else 0
            if score > best_score:
                best, best_score = (left, right), score
        if not best:
            return [tok]
        left, right = best
        out = []
        for part in (left, right):
            out.extend([part] if self._known(key(part)) else self._split(part, depth - 1))
        return out

    def _repair(self, toks):
        if _NO_GLUE:
            return toks
        # разрывы: «одинна дцать», «мик рорайон», «девя носто»
        merged, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and not toks[i].isdigit() and not toks[i + 1].isdigit():
                a, b = toks[i], toks[i + 1]
                cat = a + b
                kc = key(cat)
                a_ok = self._known(key(a)) or word_to_num(a) is not None
                b_ok = self._known(key(b)) or word_to_num(b) is not None
                if (not (a_ok and b_ok)) and (kc in NUM_BY_KEY or kc in LONG_TYPE_KEYS
                                             or kc in TYPE_BY_KEY or kc in FILLERS):
                    merged.append(cat); i += 2; continue
            merged.append(toks[i]); i += 1
        out = []
        for t in merged:
            out.extend(self._split(t))
        return out

    # ------------------------------------------------------------ основной разбор
    def parse(self, raw: str) -> Query:
        toks = self._repair(self._cut_tail(merge_ordinals(tokenize(raw))))

        cls = [self._classify(t) for t in toks]
        # ---- группа дома с конца
        i = len(toks) - 1
        while i >= 0 and cls[i] == "filler":
            i -= 1
        end = i
        while i >= 0 and cls[i] in ("num", "numw", "letter", "korp", "dom", "filler", "fl_letter"):
            i -= 1
        start = i + 1
        while start <= end and cls[start] not in ("num", "numw", "dom"):
            start += 1
        group = list(range(start, end + 1)) if start <= end else []
        if group and not any(cls[j] in ("num", "numw") for j in group):
            group = []
        # «дом» внутри группы: числа левее маркера относятся к улице
        # («квартал сто десять дом один корпус два»)
        doms = [j for j in group if cls[j] == "dom"]
        if doms and any(cls[j] in ("num", "numw") for j in group if j < doms[-1]) \
                and any(cls[j] in ("num", "numw") for j in group if j > doms[-1]):
            group = [j for j in group if j > doms[-1]]
        house, alts = self._house(toks, cls, group)
        street_idx = range(0, group[0]) if group else range(len(toks))

        q = Query(raw=raw, house=house, house_alts=alts)
        words, pending_nums = [], []
        idx = list(street_idx)
        j = 0
        while j < len(idx):
            t, c = toks[idx[j]], cls[idx[j]]
            if c == "numw":
                pending_nums.append(word_to_num(t)); j += 1; continue
            if pending_nums:
                words.extend(str(n) for n in compose(pending_nums)); pending_nums = []
            if c in ("filler", "fl_letter", "dom"):
                j += 1; continue
            if c == "type":
                q.street_type = q.street_type or self._type_of(t); j += 1; continue
            city = self._city(toks, idx, j)
            if city:
                q.city_mention = q.city_mention or city[0]
                j += city[1]; continue
            if len(t) == 1 and not t.isdigit():
                j += 1; continue          # инициалы и одиночные буквы
            words.append(t); j += 1
        if pending_nums:
            words.extend(str(n) for n in compose(pending_nums))
        # Улицы без слов не бывает. Если на улицу ничего не осталось, а в группе дома
        # два числа подряд, первое число это название: «квартал 110 1к2».
        # Только для цифр: «сто десять один» словами однозначно не делится.
        nums = [j for j in group if cls[j] in ("num", "numw")]
        if not words and len(nums) >= 2 and cls[group[0]] == "num" and toks[group[0]].isdigit():
            words = [toks[group[0]]]
            q.house, q.house_alts = self._house(toks, cls, group[1:])
        q.street_tokens = words
        q.n_words = len(words)
        return q

    @staticmethod
    def _cut_tail(toks):
        """Отрезает «кв 12», «подъезд 3», если перед маркером уже было число."""
        seen_num = False
        for i, t in enumerate(toks):
            if t in _TAIL_RAW and seen_num:
                return toks[:i]
            if t[:1].isdigit() or word_to_num(t, fuzzy=False) is not None:
                seen_num = True
        return toks

    def _classify(self, t):
        k = key(t)
        if _ORDINAL.match(t):
            return "word"
        if _HOUSE_TOKEN.match(t):
            return "num"
        if is_korpus(t):
            return "korp"
        if t in DOM_RAW:
            return "dom"
        if t in LETTERS:
            return "fl_letter" if k in FILLERS else "letter"
        if k in FILLERS:
            return "filler"
        if k in NUM_BY_KEY:
            return "numw"
        if k not in self.street_keys and word_to_num(t) is not None:
            return "numw"
        if self._type_of(t):
            return "type"
        return "word"

    def _house(self, toks, cls, group):
        items = []
        meaningful = [j for j in group if cls[j] not in ("filler", "dom")]
        for pos, j in enumerate(group):
            t, c = toks[j], cls[j]
            prev = items[-1][0] if items else None
            if c == "num":
                m = _HOUSE_TOKEN.match(t)
                items.append(("num", int(m.group(1)), False))
                if m.group(2):
                    suf = m.group(2)
                    if m.group(3):                      # «101к5»
                        items.append(("korp",)) if suf == "к" else items.append(("let", suf, suf))
                        items.append(("num", int(m.group(3)), False))
                    else:
                        items.append(("let", suf, suf))
            elif c == "numw":
                items.append(("num", word_to_num(t), True))
            elif c == "korp":
                is_last = j == meaningful[-1] if meaningful else True
                if key(t) == key("к") and is_last:
                    if prev in ("num", "korp"):
                        items.append(("let", "к", "к"))
                else:
                    items.append(("korp",))
            elif c in ("letter", "fl_letter"):
                if prev in ("num", "korp"):
                    items.append(("let", LETTERS[t], t))
        if not items:
            return "", ()
        main = render_house([it[:2] if it[0] == "let" else it for it in items])
        # «ка» в голосе чаще буква К, в чате корпус А. «к» в конце: буква или обрыв фразы
        alt_items = []
        for it in items:
            if it[0] == "let" and it[2] == "ка":
                alt_items += [("korp",), ("let", "а")]
            elif it[0] == "let" and it[2] == "к":
                continue
            else:
                alt_items.append(it[:2] if it[0] == "let" else it)
        alt = render_house(alt_items)
        return main, tuple(a for a in {alt} if a and a != main)

    def _city(self, toks, idx, j):
        for span in (2, 1):
            if j + span > len(idx):
                continue
            cand = " ".join(toks[idx[j + s]] for s in range(span))
            if cand in self.city_slang:
                return self.city_slang[cand], span
            k = key(cand)
            if k in self.city_keys:
                return self.city_keys[k], span
            if len(k) >= 7:
                # только опечатка той же или меньшей длины: «красноярский» длиннее
                # «красноярск» и остаётся частью улицы
                for ck, c in self.city_keys.items():
                    if len(ck) >= 7 and len(k) <= len(ck) and Levenshtein.distance(k, ck) <= 1:
                        return c, span
        return None
