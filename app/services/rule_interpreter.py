"""Understanding what the clinic's admin meant, not storing what they typed.

An admin's "//: shanba kuni ishlamaymiz" used to become a standing rule
word for word, and the assistant read those words in front of every reply.
Words typed on a phone between two other things are not a rule: they leave
out who it applies to and from when, they do not say what follows from them
(a closed Saturday means no Saturday appointments, not just a sentence to
repeat), and they say nothing about the rule they quietly replace. So the
model reads the instruction first, once, and writes down what it means:

* rules -- how the assistant behaves, each written to stand alone, because
  the assistant never sees the admin's message, only the rule;
* facts -- what a patient may ask and the assistant should answer, which
  belong in the knowledge base where questions can find them;
* removals -- the existing rules this cancels or replaces, so a new opening
  time does not sit beside the old one and leave the assistant to choose.

It can also ask, when the answer would change what gets written ("which
Saturday?"), or show the rules as they stand. What it may not do is write
a rule that lets the assistant diagnose, name a medicine, or invent
something: an instruction reaches here from a phone, and whoever holds that
account is not allowed to be one sentence away from that.

Only interprets. Applying the result is app.services.admin_commands.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.rag.llm import ChatMessage, LLMProvider

MAX_RULE_LENGTH = 400
MAX_FACT_ANSWER_LENGTH = 600
MAX_RULES_PER_MESSAGE = 5
MAX_FACTS_PER_MESSAGE = 3

KINDS = ("rules", "list", "clarify", "refused", "not_an_instruction")

_SYSTEM = """\
You edit the standing rules of a clinic's patient-facing front-desk \
assistant. The clinic's own staff -- its admin or its doctor -- has just sent \
the assistant an instruction. Work out what they MEAN and turn it into \
changes to the rules.

The assistant reads the rules before every patient reply and never sees the \
staff message itself, so every rule must make complete sense on its own.

Right now it is {now} in the clinic's time zone.

CURRENT RULES
{rules}

First decide what the message is (field "kind"):
- "rules": it changes how the assistant behaves or what it tells patients, \
or it cancels something set earlier.
- "list": they ask to see the current rules.
- "clarify": one missing detail would change what you write (which day, \
which service, until when, for whom). Ask ONE short question in "question". \
Never guess a detail that matters.
- "refused": it asks the assistant to diagnose, to name a medicine or a \
dose, to say it is a doctor, or to state something the clinic did not \
say. Explain in "summary" why not.
- "not_an_instruction": greeting, test message, or a question that is not \
about how the assistant should behave.

Writing rules ("rules"):
- Understand the intent, then write what the assistant must DO: when it \
applies, what to do, what to avoid. Draw the consequences a sensible \
colleague would: "we are closed on Saturdays" also means do not offer or \
accept Saturday times and suggest another day.
- Keep every concrete detail: days, hours, numbers, names, dates, \
conditions, exceptions. Add none. Do not widen or narrow the scope, and do \
not invent alternatives the staff did not give (a specific other day, a \
price, a name): say "boshqa kunni taklif qiling", not which day.
- Convert "today", "tomorrow", "this week" into absolute dates using the \
date above, so the rule still means the same tomorrow.
- Uzbek, Latin script, imperative ("... qilmang", "... ayting"), at most \
{max_rule} characters each. Not a copy of the staff's words. One rule per \
separate idea; never split one idea in two.
- If an existing rule already covers it, write nothing new and say so in \
"summary".
- If it contradicts or updates an existing rule, put that rule's number in \
"remove" and write the new one. If it only cancels a rule ("3-qoidani \
o'chir", "endi shanba ham ishlaymiz" against a closed-Saturday rule), remove \
it and write nothing in its place.

Facts ("facts"): something a patient may ask and the assistant should answer \
-- a price, an address, a new service, a schedule. "question" as a patient \
would ask it, "answer" as the clinic would say it, complete, at most \
{max_answer} characters. Facts are information; rules are behaviour. One \
instruction can give both.

Reply with ONLY this JSON, nothing else:
{{"kind": "...", "summary": "...", "rules": [], "remove": [], \
"facts": [{{"question": "...", "answer": "..."}}], "question": ""}}
"summary": one or two short Uzbek sentences, under 200 characters in \
total, telling the staff member what you understood, in your own words -- \
never a copy of theirs. When you refuse, say only why.
Removing a rule is done ONLY through "remove": a summary that says a rule \
was cancelled while "remove" is empty cancels nothing.

Examples.
Message: "bu shanba ishlamaymiz, doktor kasal"  (today is 2026-09-21)
{{"kind": "rules", "summary": "26-sentabr shanba kuni klinika yopiq, doktor \
kasal.", "rules": ["2026-09-26 (shanba) kuni klinika yopiq, doktor kasal. \
Bu kunga vaqt taklif qilmang va yozmang; so'rashsa, shu kuni ishlamasligini \
ayting va boshqa kunni taklif qiling."], "remove": [], "facts": [], \
"question": ""}}
Message: "UZI narxi endi 200 ming"
{{"kind": "rules", "summary": "UZI narxi 200 000 so'm bo'ldi.", "rules": [], \
"remove": [], "facts": [{{"question": "UZI qancha turadi?", "answer": "UZI \
narxi 200 000 so'm."}}], "question": ""}}
Message: "1-qoidani o'chir"  (CURRENT RULES: 1. Har doim shanba qabulini \
eslatib turing. 2. EKO qilmaymiz.)
{{"kind": "rules", "summary": "Shanba qabulini eslatish haqidagi 1-qoida \
bekor qilindi.", "rules": [], "remove": [1], "facts": [], "question": ""}}
Message: "yakshanba ham ishlaymiz"
{{"kind": "clarify", "summary": "", "rules": [], "remove": [], "facts": [], \
"question": "Faqat shu yakshanbami yoki doim yakshanba ham ishlaysizmi?"}}"""


class InterpretationError(Exception):
    """The model's answer could not be used."""


@dataclass(frozen=True)
class Fact:
    question: str
    answer: str


@dataclass(frozen=True)
class Interpretation:
    kind: str
    summary: str = ""
    rules: tuple[str, ...] = ()
    # Numbers into the list the model was shown, 1-based.
    remove: tuple[int, ...] = ()
    facts: tuple[Fact, ...] = ()
    question: str = ""
    # False when the model could not be used and the admin's own words stand
    # in for a rule. Said to the admin, because "understood" would be a lie.
    understood: bool = True


def verbatim(text: str) -> Interpretation:
    """The admin's own words as the rule, for when nothing better is possible.

    An instruction must never be lost because the model was down.
    """
    return Interpretation(kind="rules", rules=(text.strip()[:MAX_RULE_LENGTH],), understood=False)


def _numbered(rules: Sequence[str]) -> str:
    if not rules:
        return "(none yet)"
    return "\n".join(f"{number}. {rule}" for number, rule in enumerate(rules, start=1))


def build_prompt(existing: Sequence[str], now: datetime) -> str:
    return _SYSTEM.format(
        now=f"{now:%A %Y-%m-%d %H:%M}",
        rules=_numbered(existing),
        max_rule=MAX_RULE_LENGTH,
        max_answer=MAX_FACT_ANSWER_LENGTH,
    )


def _texts(value: object, limit: int, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned = [item.strip()[:limit] for item in value if isinstance(item, str) and item.strip()]
    return tuple(cleaned[:maximum])


def parse(raw: str, existing_count: int) -> Interpretation:
    """The model's reply as an Interpretation, or InterpretationError.

    Lenient about the wrapping -- models add code fences and a sentence
    before the JSON -- and strict about the content: an unknown kind, a
    removal of a rule that does not exist, or a "rules" answer that changes
    nothing is an error rather than a guess.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise InterpretationError("no JSON in the reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise InterpretationError("the JSON does not parse") from error
    if not isinstance(data, dict):
        raise InterpretationError("the JSON is not an object")

    kind = data.get("kind")
    if kind not in KINDS:
        raise InterpretationError(f"unknown kind {kind!r}")
    summary = data.get("summary")
    question = data.get("question")

    remove: list[int] = []
    for number in data.get("remove") or []:
        if not isinstance(number, int) or isinstance(number, bool) or number in remove:
            continue
        if not 1 <= number <= existing_count:
            raise InterpretationError(f"rule {number} does not exist")
        remove.append(number)

    facts: list[Fact] = []
    for item in data.get("facts") or []:
        if not isinstance(item, dict):
            continue
        fact_question, fact_answer = item.get("question"), item.get("answer")
        if not (isinstance(fact_question, str) and isinstance(fact_answer, str)):
            continue
        if fact_question.strip() and fact_answer.strip():
            facts.append(
                Fact(fact_question.strip()[:300], fact_answer.strip()[:MAX_FACT_ANSWER_LENGTH])
            )

    interpretation = Interpretation(
        kind=kind,
        summary=summary.strip() if isinstance(summary, str) else "",
        rules=_texts(data.get("rules"), MAX_RULE_LENGTH, MAX_RULES_PER_MESSAGE),
        remove=tuple(remove),
        facts=tuple(facts[:MAX_FACTS_PER_MESSAGE]),
        question=question.strip() if isinstance(question, str) else "",
    )
    if kind == "clarify" and not interpretation.question:
        raise InterpretationError("clarify without a question")
    changes_nothing = not (interpretation.rules or interpretation.remove or interpretation.facts)
    # "Already covered" is a legitimate answer, and it has a summary.
    if kind == "rules" and changes_nothing and not interpretation.summary:
        raise InterpretationError("rules that change nothing")
    return interpretation


async def interpret(
    text: str,
    existing: Sequence[str],
    provider: LLMProvider,
    *,
    now: datetime,
) -> Interpretation:
    """What the admin's message means, given the rules already in force."""
    reply = await provider.generate(
        build_prompt(existing, now), [ChatMessage(role="user", content=text)]
    )
    return parse(reply, len(existing))
