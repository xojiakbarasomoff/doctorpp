"""What shape a patient's message is, before anything is retrieved for it.

There is one shape that retrieval cannot survive, and it is one of the most
common things a patient types: a price question with no service in it.

    "narxi qancha"    "qancha turadi"    "сколько стоит"

The knowledge base is 3522 rows generated from one template -- "<service>
narxi qancha?" -- so a query that is only the template matches the template,
not a subject, and the nearest neighbour is whichever service has the
shortest name. Measured against production:

    ? narxi qancha              ? qancha turadi
      0.2091  mis narxi qancha?   0.0963  mis qancha turadi?
      0.2610  natriy ...          0.1785  magniy ...
      0.2704  rux ...             0.1804  temir o ...
      0.2823  MNO ...             0.1935  pertubatsiya ...
      0.2902  magniy ...          0.2007  HCA ...

0.0963 is a very confident match, so no distance cutoff will catch this: by
the only measure retrieval has, "mis qancha turadi?" really is what was
asked. The assistant then did exactly what the prompt tells it to -- answer
from the rows it was given -- and asked a patient whether they meant copper,
magnesium, iron, pertubation or HCA.

No prompt rule fixes that, because the prompt is not what is wrong. What is
wrong is the context, so this is decided here, before retrieval runs, and the
model is simply not handed the rows.

Deliberately narrow. It fires only when a message is a price question AND
nothing is left of it once the asking-about-price words and the filler are
removed. "buyrak UZI narxi" keeps "buyrak" and "uzi" and is answered
normally; "qabulga yozilmoqchiman" is not a price question at all and is
answered by the telephone rule as before.
"""

import re

# Asking what something costs, in the three languages and both alphabets this
# deployment receives. Substrings, not whole words: Uzbek agglutinates, so
# "narxi", "narxlari", "narxini" are all one marker.
_PRICE_MARKERS = (
    "narx",
    "qancha",
    "qanch",
    "turadi",
    "necha pul",
    "nechpul",
    "pul",
    "qiymat",
    "нарх",
    "қанча",
    "канча",
    "туради",
    "неча пул",
    "пул",
    "стоит",
    "стоимост",
    "сколько",
    "цена",
    "цену",
    "price",
    "cost",
)

# Words that carry no subject of their own. A message made only of these plus
# a price marker has not named anything to price.
#
# "xizmat" is here on purpose: "xizmat qancha turadi" is exactly as unanswerable
# as "qancha turadi", and a patient who writes it wants to be asked which one.
_FILLERS = {
    "salom",
    "assalom",
    "assalomu",
    "alaykum",
    "iltimos",
    "ayting",
    "aytasiz",
    "aytasizmi",
    "aytsangiz",
    "aytib",
    "bering",
    "bormi",
    "bor",
    "mumkinmi",
    "mumkin",
    "men",
    "menga",
    "siz",
    "sizda",
    "sizlarda",
    "bizda",
    "bu",
    "uchun",
    "ekan",
    "edi",
    "ha",
    "yoq",
    "ok",
    "xizmat",
    "xizmatlar",
    "xizmatingiz",
    "xizmatlaringiz",
    "salomlar",
    "яхши",
    "салом",
    "ассалом",
    "алайкум",
    "илтимос",
    "айтинг",
    "борми",
    "хизмат",
    "хизматлар",
    "менга",
    "сизда",
    "бу",
    "учун",
    "здравствуйте",
    "привет",
    "скажите",
    "пожалуйста",
    "подскажите",
    "мне",
    "вы",
    "это",
    "услуга",
    "услуги",
    "у",
    "вас",
    "какая",
    "какой",
    "какие",
    "каков",
    "какова",
    "что",
}

_APOSTROPHES = re.compile(r"['‘’ʻʼ`´]")
_NOT_WORD = re.compile(r"[^a-z0-9Ѐ-ӿ]+")


def _normalise(text: str) -> str:
    return _APOSTROPHES.sub("", text.lower())


def _tokens(text: str) -> list[str]:
    return [token for token in _NOT_WORD.split(_normalise(text)) if token]


def asks_a_price(text: str) -> bool:
    flat = _normalise(text)
    return any(marker in flat for marker in _PRICE_MARKERS)


def names_nothing_to_price(text: str) -> bool:
    """A price question with no service in it.

    True for "narxi qancha", "qancha turadi", "сколько стоит", "xizmat qancha";
    False for "buyrak UZI narxi", "ginekolog qabuli qancha", and for anything
    that is not asking a price at all.
    """
    if not asks_a_price(text):
        return False
    for token in _tokens(text):
        if token in _FILLERS:
            continue
        # A token that is itself the price question ("narxi", "qancha").
        if any(marker in token or token in marker for marker in _PRICE_MARKERS):
            continue
        # Digits alone name nothing -- "2 ta analiz" keeps "analiz", which is
        # what decides it.
        if token.isdigit():
            continue
        return False
    return True
