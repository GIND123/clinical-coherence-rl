"""Amendment 6: the open-ended format arm and its judge."""
import json

from ccrl.panel import openended as O


def test_vignette_extraction_roundtrip():
    from ccrl.prompts import templates as T
    v = "A 50-year-old woman has fever.\n\nShe also reports cough."
    cands = [("AA", "Pneumonia"), ("AB", "Asthma")]
    assert O.vignette_of(T.user_prompt(v, cands)) == v
    assert O.vignette_of(T.user_prompt(v, None)) == v


def test_open_validator():
    ok, inf, js = O.open_validator('{"differential": ["Pneumonia", "Asthma", "COPD", "Bronchitis", "Influenza"]}')
    assert ok and inf and json.loads(js)["differential"][0] == "Pneumonia"
    ok, inf, js = O.open_validator('```json\n{"differential": ["Pneumonia", "Asthma"]}\n```')
    assert ok and not inf                     # parsed, but fewer than five
    assert O.open_validator('{"differential": []}')[0] is False
    assert O.open_validator("Pneumonia")[0] is False
    ok, _, js = O.open_validator('{"differential": ["a","b","c","d","e","f","g"]}')
    assert ok and len(json.loads(js)["differential"]) == 5


def test_answer_ids_hide_the_base_rendering():
    firsts = {O.answer_ids(f"c{i}", "glm-5", O.JOB)["base"] for i in range(40)}
    assert firsts == set("ABCD")              # the base answer is not always "A"
    a = O.answer_ids("c1", "glm-5", O.JOB)
    assert sorted(a.values()) == list("ABCD") and a == O.answer_ids("c1", "glm-5", O.JOB)


def test_judge_check():
    ids = {"A": ["Pneumonia", "Asthma"], "B": ["Zebra fever"]}
    chk = O.judge_check(ids, ["AA", "AB"])
    assert chk({"maps": {"A": ["AA", "X1"], "B": ["X2"]}}) is None
    assert chk({"maps": {"A": ["AA"], "B": ["X2"]}}) is not None          # wrong length
    assert chk({"maps": {"A": ["AA", "ZZ"], "B": ["X2"]}}) is not None    # unknown code
    assert chk({"maps": {"A": ["AA", "AB"]}}) is not None                 # missing answer


def test_job_models():
    assert "kimi-k3" in O.job_models(O.JOB) and "kimi-k3" not in O.job_models(O.JOB_T0)
    assert len(O.job_models(O.JOB_T0)) == len(O.job_models(O.JOB)) - 1


def test_mock_judge_maps_names():
    from ccrl.bedrock.mock import MockBedrock
    user = ("Candidate diagnoses:\nAA  Pneumonia\nAB  Asthma\n\nAnswers (id: list of diagnoses, most likely first):\n"
            + json.dumps({"A": ["pneumonia", "Zebra fever"], "B": ["Zebra fever"]}) + "\n\nReturn the JSON object described above.")
    out = json.loads(MockBedrock._open_judge(user))
    assert out["maps"]["A"][0] == "AA" and out["maps"]["A"][1] == out["maps"]["B"][0] == "X1"
