import pandas as pd

from ccrl.train import general as G


def test_parsers():
    assert G.parse_mmlu("C", 10) == "C"
    assert G.parse_mmlu("The answer is (D).", 10) == "D"
    assert G.parse_mmlu("J", 4) == ""
    assert G.gsm_gold("so 3 + 4 = 7\n#### 1,234") == "1234"
    assert G.parse_gsm("work...\n#### 42") == "42"
    assert G.parse_gsm("the total is 12.50 dollars") == "12.5"
    assert G.parse_gsm("") == ""


def test_mmlu_selection_is_seeded_and_stratified():
    df = pd.DataFrame({"question_id": range(1000), "category": ["a"] * 700 + ["b"] * 300})
    s1, s2 = G.select_mmlu(df, 100), G.select_mmlu(df.sample(frac=1, random_state=3), 100)
    assert list(s1["question_id"]) == list(s2["question_id"])
    assert len(s1) == 100 and (s1["category"] == "a").sum() == 70


def test_noninferior():
    base = pd.DataFrame({"task": "gsm8k", "qid": range(400), "correct": [1.0, 0.0] * 200})
    same = base.copy()
    rows = G.noninferior(same, base, "x")
    assert rows[0]["supported"] and rows[0]["diff"] == 0
    worse = base.copy()
    worse.loc[:79, "correct"] = 0.0
    assert not G.noninferior(worse, base, "x")[0]["supported"]
