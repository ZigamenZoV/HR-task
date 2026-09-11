"""Числительные словами -> цифры и разбор номера дома.

Правило склейки чисел: следующее число «встаёт» в свободный разряд текущего
(«сорок» + «четыре» = 44, «триста» + «восемьдесят» + «один» = 381). Если разряд
занят («три один», «двадцать один пять»), второе число считается корпусом:
3к1, 21к5. Именно так в данных устроены голосовые ответы.
"""
import re
from .text import key

_WORDS = {
    0: "ноль", 1: "один одна одно", 2: "два две", 3: "три", 4: "четыре", 5: "пять",
    6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
    11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
    15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
    19: "девятнадцать", 20: "двадцать", 30: "тридцать", 40: "сорок",
    50: "пятьдесят", 60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят",
    90: "девяносто", 100: "сто", 200: "двести", 300: "триста", 400: "четыреста",
    500: "пятьсот", 600: "шестьсот", 700: "семьсот", 800: "восемьсот", 900: "девятьсот",
}
NUM_BY_KEY = {key(w): n for n, ws in _WORDS.items() for w in ws.split()}

# Буквы корпуса, как их произносят. Сравнение по сырому тексту: фонетический ключ
# склеил бы «бэ» и «пэ», «гэ» и «к». «э» не буква, а заминка.
LETTERS = {w: l for l, ws in {
    "а": "а", "б": "б бэ бе", "в": "в вэ ве", "г": "г гэ ге", "д": "д дэ де",
    "е": "е", "и": "и", "к": "к ка", "п": "п пэ пе",
}.items() for w in ws.split()}

# Короткие маркеры сравниваем по сырому тексту (ключ «г» совпал бы с «к»),
# длинные по ключу, чтобы ловить «тробь», «карпус».
KORPUS_RAW = {"к", "/", "кор", "корп", "корпус", "стр", "строение"}
KORPUS_KEYS = {key(w) for w in "корпус карпус дробь".split()}
DOM_RAW = {"д", "дом"}


def is_korpus(t: str) -> bool:
    return t in KORPUS_RAW or (len(t) >= 4 and key(t) in KORPUS_KEYS)

_HOUSE_TOKEN = re.compile(r"^(\d+)([а-я]{0,2})(\d*)$")


def _edit1(a: str, b: str) -> bool:
    """Расстояние Левенштейна <= 1 без внешних зависимостей."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1; continue
        diff += 1
        if diff > 1:
            return False
        if la > lb: i += 1
        elif lb > la: j += 1
        else: i += 1; j += 1
    return diff + (la - i) + (lb - j) <= 1


def word_to_num(tok: str, fuzzy: bool = True):
    """Число по слову. Нечёткий режим (одна правка в ключе) только для длинных слов:
    на коротких «ясная» превращалась в «одна»."""
    k = key(tok)
    if k in NUM_BY_KEY:
        return NUM_BY_KEY[k]
    if fuzzy and len(k) >= 5:
        for nk, n in NUM_BY_KEY.items():
            if len(nk) >= 5 and _edit1(k, nk):
                return n
    return None


def compose(nums):
    """[40, 4] -> [44]; [3, 1] -> [3, 1]; [300, 80, 1] -> [381]."""
    out, cur = [], None
    for n in nums:
        if cur is None:
            cur = n
        elif cur >= 100 and cur % 100 == 0 and n < 100:
            cur += n
        elif cur >= 20 and cur % 10 == 0 and n < 10:
            cur += n
        else:
            out.append(cur)
            cur = n
    if cur is not None:
        out.append(cur)
    return out


def split_house(h: str):
    """Каноника дома: '101К5' -> (101, 'к5'). Суффикс: всё после основного номера."""
    h = h.lower().replace(" ", "").replace("/", "к")
    m = re.match(r"^(\d+)(.*)$", h)
    if not m:
        return None, h
    return int(m.group(1)), m.group(2)


def render_house(items):
    """items: ('num', int, from_words) | ('let', str) | ('korp',) -> каноника '381к3'."""
    merged, buf = [], []
    for it in items:                       # склеиваем подряд идущие числа, сказанные словами
        if it[0] == "num" and it[2]:
            buf.append(it[1]); continue
        if buf:
            merged.extend(("num", n) for n in compose(buf)); buf = []
        merged.append(it[:2] if it[0] == "num" else it)
    if buf:
        merged.extend(("num", n) for n in compose(buf))
    parts, seen_num, last = [], False, None
    for it in merged:
        if it[0] == "num":
            if seen_num and last == "num":
                parts.append("к")          # «три один» -> 3к1
            parts.append(str(it[1])); seen_num = True
        elif it[0] == "korp":
            if last == "korp":
                continue
            parts.append("к")
        else:
            parts.append(it[1])
        last = it[0]
    return "".join(parts)
