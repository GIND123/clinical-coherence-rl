import pytest

from ccrl.panel import items as I
from ccrl.workspace import get as get_ws


@pytest.fixture()
def built(tmp_runs):
    ws = get_ws(True)
    s1 = I.build(ws, allow_missing_para=True, allow_short_mcr=True, verbose=False)
    return ws, s1


def test_deterministic(built):
    ws, s1 = built
    s2 = I.build(ws, allow_missing_para=True, allow_short_mcr=True, verbose=False)
    assert s1["items_sha256"] == s2["items_sha256"]


def test_transform_semantics(built):
    ws, _ = built
    items = I.load_items(ws)
    sets = I.load_sets(ws)
    assert len([i for i in sets["panel"] if i.startswith("ddx")]) == 3000
    cid = items[sets["panel"][0]]["case_id"]
    g = {t: items[f"{cid}::{t}"] for t in
         ("base", "perm1", "perm2", "perm3", "dup", "cue", "authority", "ablate", "retest")}
    assert g["base"]["turns"] == g["retest"]["turns"]
    orders = {tuple(g[t]["meta"]["order"]) for t in ("base", "perm1", "perm2", "perm3")}
    assert len(orders) == 4                                   # four distinct presentations
    for t in ("perm1", "perm2", "perm3"):
        assert g[t]["meta"]["order"][0] == g["base"]["meta"]["order"][0]          # chief complaint fixed
        assert sorted(g[t]["meta"]["order"]) == sorted(g["base"]["meta"]["order"])  # same SET
    assert "As noted above" in g["dup"]["turns"][0]
    assert len(g["ablate"]["meta"]["order"]) == len(g["base"]["meta"]["order"]) - 1
    assert g["ablate"]["meta"]["ablated"] not in g["ablate"]["meta"]["order"]
    assert "50%" in g["authority"]["turns"][0]
    assert g["authority"]["meta"]["suggested"] != g["base"]["gold"]
    assert abs(g["authority"]["meta"]["normative_log_lr"] - 3.871201) < 1e-5      # log 48 for K = 49


def test_sequential_items(built):
    ws, _ = built
    items = I.load_items(ws)
    seq = I.load_sets(ws)["sequential"]
    a = items[next(i for i in seq if i.endswith("seqA"))]
    b = items[f"{a['case_id']}::seqB"]
    assert len(a["turns"]) == 2 and len(b["turns"]) == 2
    assert set(a["meta"]["first"][1:]) == set(b["meta"]["second"])            # halves swapped
