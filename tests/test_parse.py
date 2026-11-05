import numpy as np

from ccrl.prompts.parse import parse_posterior

CODES = ["AA", "AB", "AC"]


def test_plain():
    r = parse_posterior('{"dx": "AB", "p": {"AA": 20, "AB": 70, "AC": 10}}', CODES)
    assert r.ok and r.dx == "AB" and np.allclose(r.probs, [0.2, 0.7, 0.1]) and r.informative


def test_fence_think_and_last_object_wins():
    t = '<think>draft {"dx": "AA", "p": {"AA": 100}}</think>```json\n{"dx": "AC", "p": {"AA": 0, "AB": 0, "AC": 100}}\n```'
    r = parse_posterior(t, CODES)
    assert r.ok and r.dx == "AC" and r.probs[2] == 1.0


def test_all_zero_is_uniform_not_failure():
    r = parse_posterior('{"dx": "AA", "p": {"AA": 0, "AB": 0, "AC": 0}}', CODES)
    assert r.ok and r.all_zero and not r.informative and np.allclose(r.probs, 1 / 3)


def test_missing_unknown_and_sum():
    r = parse_posterior('{"dx": "AA", "p": {"AA": "60%", "AB": 30, "ZZ": 5}}', CODES)
    assert r.ok and r.n_missing == 1 and r.n_unknown == 1 and "sum_not_100" in r.flags
    assert np.allclose(r.probs, [60 / 90, 30 / 90, 0])


def test_failures():
    assert not parse_posterior("no json here", CODES).ok
    assert not parse_posterior('{"dx": "AA", "p": {"AA": -5, "AB": 10}}', CODES).ok
    assert not parse_posterior('{"dx": "AA", "p": {"AA": "high"}}', CODES).ok
    assert not parse_posterior('{"dx": "AA", "p": {"QQ": 10}}', CODES).ok


def test_names_as_keys_and_scores_key():
    names = {"AA": "Flu", "AB": "Cold", "AC": "Asthma"}
    r = parse_posterior('{"dx": "Cold", "p": {"Flu": 10, "Cold": 90, "Asthma": 0}}', CODES, names=names)
    assert r.ok and r.dx == "AB"
    r = parse_posterior('{"dx": "AA", "s": {"AA": 80, "AB": 80, "AC": 40}}', CODES, key="s")
    assert r.ok and np.allclose(r.probs, [0.4, 0.4, 0.2])
