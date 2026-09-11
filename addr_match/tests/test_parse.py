"""Регрессионные тесты на строках из данных. Каждый кейс когда-то ломался.

python -m pytest -q tests        (или просто: python tests/test_parse.py)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from addrmatch.matcher import Matcher      # noqa: E402
from addrmatch.text import key             # noqa: E402
from addrmatch import metrics              # noqa: E402

M = Matcher.load(os.path.join(ROOT, "data", "etalon.jsonl"), os.path.join(ROOT, "models", "model.json"))
P = M.parser


def parse(s):
    q = P.parse(s)
    return q.street_tokens, q.house


def test_phonetic_key_folds_voice_noise():
    assert key("пирфомайский") == key("первомайский")
    assert key("краснозвиздная") == key("краснозвёздная")
    assert key("киравсгой") == key("кировской")


def test_house_words_and_korpus():
    assert parse("улица щорса триста восемьдесят один к три")[1] == "381к3"
    assert parse("улица щорса три один")[1] == "3к1"              # занятый разряд = корпус
    assert parse("ну я на улица люблинская 72 трабь 3 нахожусь")[1] == "72к3"
    assert parse("улица лавочкина три кдва")[1] == "3к2"          # склейка «к» + «два»


def test_letters_by_raw_text():
    # через фонетический ключ «бэ» превращалась в «п», «г» в «к»
    assert parse("проезд булатниковский шестнадцать б так")[1] == "16б"
    assert parse("десять пэ")[1] == "10п"


def test_short_words_are_not_numbers():
    # нечёткое «ясная» ~ «одна» давало дом 1к31
    assert parse("ясная 31") == (["ясная"], "31")


def test_house_is_taken_from_the_end():
    assert parse("улица 9 мая 12") == (["9", "мая"], "12")


def test_glue_and_breaks():
    q = P.parse("шоссемосковское одинна дцать")
    assert (q.street_tokens, q.house, q.street_type) == (["московское"], "11", "шоссе")
    assert P.parse("мик рорайон вот первомайский 23").street_type == "микрорайон"
    assert parse("ленина54") == (["ленина"], "54")


def test_apartment_tail_is_cut():
    # в данных такого нет, но в живом трафике «кв 12» после дома встретится
    assert parse("пр-т ленина 28 кв 12") == (["ленина"], "28")
    assert parse("проспект ленина двадцать восемь квартира двенадцать") == (["ленина"], "28")
    assert parse("ленина 28 подъезд 3") == (["ленина"], "28")   # без обрезки «подъезд» ~ «пятьдесят»


def test_city_mention_is_strict():
    assert P.parse("красноярск улица мира 5").city_mention == "Красноярск"
    assert P.parse("проспект красноярский рабочий 54").city_mention == ""


def test_end_to_end():
    addr = lambda ids: M.idx.by_id[ids[0]]["address"] if ids else None
    assert addr(M.match("живу касданаевская 63 корпус 2", "Москва")[0]) == \
        "г. Москва, ул. Кастанаевская, д. 63к2"
    assert addr(M.match("мне удабно э на акатемика бардены дом одиннатцать к а", "Екатеринбург")[0]) == \
        "г. Екатеринбург, ул. Академика Бардина, д. 11кА"
    assert M.match("да куда скажете туда и приду", "Иркутск") == ([], [])
    assert M.match("Бунбова 56", "Иваново") == ([], [])


def test_labeled_metrics_do_not_regress():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "adresses_labeled.jsonl"), encoding="utf-8")]
    preds = {r["id"]: M.match(r["raw_adress"], r["city"], r["channel"])[0] for r in rows}
    m = metrics.compute(rows, preds)
    assert m["top1"] >= 0.99 and m["reject_recall"] >= 0.99 and m["reject_precision"] >= 0.98, m


def test_city_slang_by_raw_text_only():
    # по фонетическому ключу «нижний» совпадал с «нижне» из Нижне-Печерской
    q = P.parse("нижне-печерская 4")
    assert (q.city_mention, q.street_tokens) == ("", ["нижне", "печерская"])
    assert P.parse("в нижнем улица родионова 5").city_mention == "Нижний Новгород"


def test_numeric_street_names():
    # улица из одних цифр: без правила «110» уходило в номер дома, получалось 110к1к2
    assert parse("квартал 110 1к2") == (["110"], "1к2")
    # «дом» внутри числовой группы отделяет улицу от номера
    assert parse("квартал сто десять дом один корпус два") == (["110"], "1к2")
    assert parse("улица ленина дом 5") == (["ленина"], "5")


def test_ordinal_same_in_query_and_etalon():
    # эталон делал из «6-й» токен «6», запрос «6й»: ratio 0.67 и отказ
    ids, _ = M.match("6-й 3ка", "Нижний Новгород")
    assert ids and M.idx.by_id[ids[0]]["address"] == "г. Нижний Новгород, мкр 6-й, д. 3кА"


def test_twins_in_two_cities_need_a_city():
    # «Лескова 21» есть в Новосибирске и Нижнем Новгороде: без города это монетка
    assert M.match("лескова 21", "") == ([], [])
    ids, _ = M.match("лескова 21", "Новосибирск")
    assert M.idx.by_id[ids[0]]["city"] == "Новосибирск"


def test_whole_etalon_self_retrieval():
    # каждый адрес эталона как запрос; единственное исключение: «10комната 513»
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "etalon.jsonl"), encoding="utf-8")]
    grp = lambda i: M.idx.dup_of.get(i, i)
    bad = []
    for e in rows:
        ids, _ = M.match(f"{e['street_type']} {e['street']} {e['house']}", e["city"])
        if not ids or grp(ids[0]) != grp(e["etalon_id"]):
            bad.append(e["address"])
    assert bad == ["г. Москва, ш. Калужское 22-й км, д. 10комната 513"], bad


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
