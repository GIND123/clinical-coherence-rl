"""Render DDXPlus finding units as clinical free text.

DDXPlus stores findings as questionnaire items ("Do you have a fever?").
This module turns each into a third-person verb phrase that follows
"The patient ..." ("has a fever (either felt or measured with a thermometer)"),
and each finding unit (parent + descriptors) into one sentence.

Two phrase sets exist:
  * ``base``  — the deterministic rule-based phrases below (variant 0);
  * ``para``  — the frozen phrase bank (variant 1): an independently worded
    phrase per evidence code, generated once by the preprocessor model,
    checked for equivalence by the verifier model, and reviewed by
    clinicians. Value meanings (body parts, colours, 0-10 ratings, regions)
    are never reworded, so paraphrase cannot change what was observed.

Coverage is a test (tests/test_ddx_render.py): every one of the 223 released
evidence codes must render without falling through to the generic form.
"""
from __future__ import annotations

import re

from ccrl.data.ddxplus import KB, Unit

# ------------------------------------------------------------ binary phrases
_PRONOUNS = [
    (re.compile(r"\byourself\b", re.I), "themselves"),
    (re.compile(r"\byour\b", re.I), "their"),
    (re.compile(r"\byou\b", re.I), "they"),
]
_IRREG_3SG = {"have": "has", "be": "is", "do": "does", "go": "goes"}
_IRREG_PAST = {"lose": "lost", "eat": "ate", "have": "had", "be": "was", "feel": "felt",
               "take": "took", "drink": "drank", "begin": "began", "get": "got"}


def _third_person(verb: str) -> str:
    v = verb.lower()
    if v in _IRREG_3SG:
        return _IRREG_3SG[v]
    if v.endswith(("s", "sh", "ch", "x", "z")):
        return v + "es"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _past(verb: str) -> str:
    v = verb.lower()
    if v in _IRREG_PAST:
        return _IRREG_PAST[v]
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def _depersonalise(text: str) -> str:
    for pat, repl in _PRONOUNS:
        text = pat.sub(repl, text)
    return text


# (opener regex, rewrite of the remainder). First match wins.
_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"^do you have or have you ever had\b", re.I), lambda r: f"has or has had {r}"),
    (re.compile(r"^have you or any member of your family ever had\b", re.I), lambda r: f"or a family member has had {r}"),
    (re.compile(r"^are there any members of your family who have been diagnosed( with)?\b", re.I), lambda r: f"has a family member diagnosed with {r}"),
    (re.compile(r"^are there members of your family who have been diagnosed( with)?\b", re.I), lambda r: f"has a family member diagnosed with {r}"),
    (re.compile(r"^do any members of your immediate family have\b", re.I), lambda r: f"has an immediate family member with {r}"),
    (re.compile(r"^do you have any family members who have\b", re.I), lambda r: f"has a family member with {r}"),
    (re.compile(r"^are you consulting because you have\b", re.I), lambda r: f"is consulting because they have {r}"),
    (re.compile(r"^have you started or taken\b", re.I), lambda r: f"has started or taken {r}"),
    (re.compile(r"^in the last month, have you been\b", re.I), lambda r: f"in the last month has been {r}"),
    (re.compile(r"^do you have\b", re.I), lambda r: f"has {r}"),
    (re.compile(r"^do you feel\b", re.I), lambda r: f"feels {r}"),
    (re.compile(r"^do you live\b", re.I), lambda r: f"lives {r}"),
    (re.compile(r"^do you take\b", re.I), lambda r: f"takes {r}"),
    (re.compile(r"^do you suffer\b", re.I), lambda r: f"suffers {r}"),
    (re.compile(r"^do you work\b", re.I), lambda r: f"works {r}"),
    (re.compile(r"^do you regularly (\w+)\b", re.I), None),  # handled in _declarative
    (re.compile(r"^do you currently (\w+)\b", re.I), None),
    (re.compile(r"^do your\b", re.I), lambda r: f"reports that their {r}"),
    (re.compile(r"^does your\b", re.I), lambda r: f"reports that their {r}"),
    (re.compile(r"^does the\b", re.I), lambda r: f"reports that the {r}"),
    (re.compile(r"^have you ever\b", re.I), lambda r: f"has ever {r}"),
    (re.compile(r"^have you had\b", re.I), lambda r: f"has had {r}"),
    (re.compile(r"^have you been\b", re.I), lambda r: f"has been {r}"),
    (re.compile(r"^have you noticed\b", re.I), lambda r: f"has noticed {r}"),
    (re.compile(r"^have you recently\b", re.I), lambda r: f"has recently {r}"),
    (re.compile(r"^have you lost\b", re.I), lambda r: f"has lost {r}"),
    (re.compile(r"^have you traveled\b", re.I), lambda r: f"has travelled {r}"),
    (re.compile(r"^have any of\b", re.I), lambda r: f"reports that any of {r}"),
    (re.compile(r"^has any\b", re.I), lambda r: f"reports that any {r}"),
    (re.compile(r"^are you currently\b", re.I), lambda r: f"is currently {r}"),
    (re.compile(r"^are your symptoms\b", re.I), lambda r: f"has symptoms that are {r}"),
    (re.compile(r"^are your\b", re.I), lambda r: f"reports that their {r} are up to date"
     if r.strip().lower().startswith("vaccinations") else f"reports that their {r}"),
    (re.compile(r"^are the\b", re.I), lambda r: f"reports that the {r}"),
    (re.compile(r"^is your\b", re.I), lambda r: f"reports that their {r}"),
    (re.compile(r"^is the\b", re.I), lambda r: f"reports that the {r}"),
    (re.compile(r"^were you born\b", re.I), lambda r: f"was born {r}"),
    (re.compile(r"^were you\b", re.I), lambda r: f"was {r}"),
    (re.compile(r"^did your\b", re.I), lambda r: f"reports that their {r}"),
    (re.compile(r"^did you previously\b", re.I), lambda r: f"previously {r}"),
]


def _declarative(question: str) -> str | None:
    """Questionnaire item -> verb phrase following 'The patient'."""
    q = question.strip().rstrip("?").rstrip(":").strip()
    m = re.match(r"^do you (regularly|currently) (\w+)\b(.*)$", q, re.I)
    if m:
        return f"{m.group(1).lower()} {_third_person(m.group(2))}{m.group(3)}".strip()
    m = re.match(r"^did you (\w+)\b(.*)$", q, re.I)
    if m and m.group(1).lower() != "previously":
        return f"{_past(m.group(1))}{m.group(2)}".strip()
    for pat, fn in _RULES:
        if fn is None:
            continue
        if pat.search(q):
            return fn(pat.sub("", q, count=1).strip())
    m = re.match(r"^do you (\w+)\b(.*)$", q, re.I)
    if m:
        return f"{_third_person(m.group(1))}{m.group(2)}".strip()
    m = re.match(r"^are you (\w+)\b(.*)$", q, re.I)
    if m:
        return f"is {m.group(1)}{m.group(2)}".strip()
    m = re.match(r"^have you (\w+)\b(.*)$", q, re.I)
    if m:
        return f"has {m.group(1)}{m.group(2)}".strip()
    return None


# ------------------------------------------------------- non-binary phrases
# code -> template; "{value}" is filled with the rendered value(s).
# Children of E_53 (pain) and E_129 (skin) are phrased so they read naturally
# right after the parent phrase.
VALUE_TEMPLATES: dict[str, str] = {
    "E_54": "describes the pain as {value}",
    "E_55": "localizes the pain to {value}",
    "E_56": "rates the pain intensity at {value} out of 10",
    "E_57": "reports that the pain radiates to {value}",
    "E_58": "rates how precisely the pain can be located at {value} out of 10",
    "E_59": "rates how suddenly the pain appeared at {value} out of 10 (0 = very gradually, 10 = very suddenly)",
    "E_130": "describes the skin lesion colour as {value}",
    "E_132": "rates the swelling of the skin lesions at {value} out of 10",
    "E_133": "has the affected skin region located at {value}",
    "E_134": "rates the pain caused by the skin lesions at {value} out of 10",
    "E_136": "rates the itching at {value} out of 10",
    "E_152": "has swelling located at {value}",
    "E_204": "has travelled to {value} in the last 4 weeks",
}
# Yes/No categorical evidences: (phrase if Y, phrase if N)
YES_NO: dict[str, tuple[str, str]] = {
    "E_131": ("has skin lesions that peel off", "has skin lesions that do not peel off"),
    "E_135": ("has skin lesions larger than 1 cm", "has skin lesions no larger than 1 cm"),
    "E_204": ("has travelled out of the country in the last 4 weeks",
              "has not travelled out of the country in the last 4 weeks"),
}
# Special values that read badly through the template.
SPECIAL_VALUES: dict[tuple[str, str], str] = {
    ("E_57", "nowhere"): "reports that the pain does not radiate",
    ("E_55", "nowhere"): "does not localize the pain to any particular place",
    ("E_133", "nowhere"): "does not have a particular affected skin region",
    ("E_152", "nowhere"): "does not have swelling at a particular site",
}
_VALUE_FIX = {"ischio jambier": "hamstring", "Caraibes": "the Caribbean", "belly": "abdomen",
              "epigastric": "epigastric region", "a knife stroke": "like a knife stroke",
              "a cramp": "like a cramp", "a pulse": "pulsating"}
_SIDE = re.compile(r"^(.*?)\s*\((L|R)\)\s*$")


def render_value(raw: str) -> str:
    v = raw.strip()
    m = _SIDE.match(v)
    if m:
        v = f"{'left' if m.group(2) == 'L' else 'right'} {m.group(1).strip()}"
    for a, b in _VALUE_FIX.items():
        if v == a or v.endswith(" " + a):
            v = v[: len(v) - len(a)] + b
    return v


def join_values(vals: list[str]) -> str:
    vals = [v for v in vals if v]
    if len(vals) <= 1:
        return vals[0] if vals else ""
    return ", ".join(vals[:-1]) + " and " + vals[-1]


# Hand-curated phrases for compound or irregular questions, where the rule
# set above produces ungrammatical text. Reviewed against the English
# question; meaning is unchanged (typos in the release are corrected).
OVERRIDES: dict[str, str] = {
    "E_1": "is currently being treated, or was recently treated, with an oral antibiotic for an ear infection",
    "E_4": "has had croup, or has a family member who has had croup",
    "E_8": "currently undergoes dialysis",
    "E_10": "is currently taking, or has recently taken, anti-inflammatory drugs (NSAIDs)",
    "E_23": "sometimes temporarily stops breathing while asleep",
    "E_25": "has a family member diagnosed with cluster headaches",
    "E_26": "has a family member diagnosed with anemia",
    "E_40": "has been in contact with someone who has had pertussis (whooping cough)",
    "E_42": "has been in contact with, or has eaten, something they are allergic to",
    "E_43": "has lost consciousness with violent and sustained muscle contractions, or has had an absence episode",
    "E_49": "attends or works in a daycare",
    "E_53": "has pain related to their reason for consulting",
    "E_65": "has difficulty swallowing, or a feeling of discomfort or blockage when swallowing",
    "E_67": "has bouts of choking or shortness of breath that wake them up at night",
    "E_70": "is significantly overweight compared with people of the same height",
    "E_71": "has high cholesterol or takes medication to treat high cholesterol",
    "E_78": "drinks alcohol excessively or has an addiction to alcohol",
    "E_82": "feels lightheaded and dizzy, or feels about to faint",
    "E_86": "has close family members who suffer from allergies (any type), hay fever or eczema",
    "E_88": "feels so tired that they are unable to do their usual activities, or is stuck in bed all day",
    "E_89": "constantly feels fatigued or has non-restful sleep",
    "E_92": "had cheeks that suddenly turned red",
    "E_99": "has had a migraine, or has a family member known to have migraines",
    "E_104": "has high blood pressure or takes medication to treat high blood pressure",
    "E_105": "has had a heart attack or has angina (chest pain)",
    "E_111": "feels like they are dying, or was afraid that they were about to die",
    "E_112": "wheezes while inhaling, or has noisy breathing after coughing spells",
    "E_114": "is more irritable than usual, or their mood has been very unstable recently",
    "E_124": "has asthma, or has had to use a bronchodilator in the past",
    "E_128": "has at times felt like they were suffocating for a very short time, with an inability to breathe or speak",
    "E_129": "has lesions, redness or other skin problems that they believe are related to their reason for consulting",
    "E_142": "reports that their mother suffers from asthma",
    "E_146": "is taking a new oral anticoagulant (NOAC)",
    "E_148": "feels nauseous or feels like vomiting",
    "E_149": "takes a calcium channel blocker (medication)",
    "E_154": "has skin that is much paler than usual",
    "E_155": "feels their heart beating fast (racing) or irregularly (missing a beat), or has palpitations",
    "E_160": "was born prematurely or had a complication at birth",
    "E_161": "has recently had a loss of appetite, or gets full more quickly than usual",
    "E_163": "has had vaginal discharge",
    "E_165": "has a family member who has had a pneumothorax",
    "E_167": "thinks they are pregnant, or is currently pregnant",
    "E_169": "has an itchy nose or an itchy back of the throat",
    "E_174": "has been losing weight unintentionally, or has lost their appetite",
    "E_175": ("has noticed new fatigue, generalized and vague discomfort, diffuse (widespread) muscle aches "
              "or a change in their general well-being related to their reason for consulting"),
    "E_176": "previously had, or currently has, weakness or paralysis in one or more limbs or in the face",
    "E_177": "currently has, or has had, numbness, loss of sensitivity or tingling somewhere on their body",
    "E_178": "has noticed unusual bleeding or bruising related to their reason for consulting",
    "E_192": "feels that muscle spasms or soreness in their neck keep them from turning their head to one side",
    "E_202": "has a whooping cough",
    "E_205": "suddenly has difficulty or an inability to open their mouth, or has jaw pain when opening it",
    "E_207": "lives in a big city",
    "E_208": "has a BMI below 18.5, or is underweight",
    "E_209": "reports that their vaccinations are up to date",
    "E_211": "has vomited several times, or has made several efforts to vomit",
    "E_221": ("has symptoms or pain that increase with coughing, with an effort such as lifting a weight, "
              "or when forcing a bowel movement"),
}
_EVER = [("has ever had ", "has previously had "), ("has ever been ", "has previously been ")]


def base_template(kb: KB, code: str) -> str:
    """The variant-0 template for one evidence code ('{value}' if valued)."""
    ev = kb.evidences[code]
    if code in VALUE_TEMPLATES:
        return VALUE_TEMPLATES[code]
    if code in YES_NO:            # rendered through YES_NO; template unused
        return "{value}"
    if code in OVERRIDES:
        return OVERRIDES[code]
    if ev.data_type == "B":
        d = _declarative(ev.question)
        if not d:
            return f"reports: {_depersonalise(ev.question.rstrip('?'))}"
        d = _depersonalise(d)
        for a, b in _EVER:
            if d.startswith(a):
                d = b + d[len(a):]
        return d
    raise KeyError(f"no template for non-binary evidence {code}")


def all_base_templates(kb: KB) -> dict[str, str]:
    return {c: base_template(kb, c) for c in kb.evidences}


def unrendered_codes(kb: KB) -> list[str]:
    """Codes that fall through to the generic 'reports:' form (should be [])."""
    return [c for c in kb.evidences if base_template(kb, c).startswith("reports: ")]


def code_phrase(kb: KB, code: str, values: list[str], templates: dict[str, str]) -> str:
    """Verb phrase for one code with its values, using the given template set."""
    ev = kb.evidences[code]
    if ev.data_type == "B" and not values:
        return templates[code]
    meanings = [ev.value_meaning.get(v, v) for v in values]
    if code in YES_NO and len(meanings) == 1 and meanings[0] in ("Y", "N"):
        yes, no = YES_NO[code]
        alt = templates.get(f"{code}#{meanings[0]}")
        return alt or (yes if meanings[0] == "Y" else no)
    if len(meanings) == 1 and (code, meanings[0]) in SPECIAL_VALUES:
        alt = templates.get(f"{code}#{meanings[0]}")
        return alt or SPECIAL_VALUES[(code, meanings[0])]
    rendered = [render_value(m) for m in meanings if m not in ("NA", "nowhere")]
    tpl = templates[code]
    return tpl.replace("{value}", join_values(rendered))


def unit_sentence(kb: KB, unit: Unit, templates: dict[str, str]) -> str:
    """One finding unit as one sentence: parent phrase, then its descriptors."""
    head_vals = unit.values.get(unit.uid, [])
    parts = [code_phrase(kb, unit.uid, head_vals, templates)]
    for code in sorted((c for c in unit.values if c != unit.uid), key=lambda c: int(c.split("_")[1])):
        parts.append(code_phrase(kb, code, unit.values[code], templates))
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = "; ".join(parts[:-1]) + "; and " + parts[-1]
    body = body.strip()
    return f"The patient {body}."


def context_sentence(age: int, sex: str, variant: str = "base") -> str:
    if age < 1:
        who = "male infant under 1 year old" if sex == "M" else "female infant under 1 year old"
        noun = f"A {who}"
    elif age < 18:
        noun = f"A {age}-year-old {'boy' if sex == 'M' else 'girl'}"
    else:
        noun = f"A {age}-year-old {'man' if sex == 'M' else 'woman'}"
    if noun.startswith("A 8") or noun.startswith("A 11-") or noun.startswith("A 18-"):
        noun = "An" + noun[1:]
    if variant == "para":
        return f"{noun} is seen for a medical evaluation."
    return f"{noun} presents for evaluation."
