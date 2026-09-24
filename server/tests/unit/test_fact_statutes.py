from app.tools import fact_statutes as fs

IPC, CRPC, BNS = fs.IPC, fs.CRPC, fs.BNS


def ranked(per_window, **kw):
    return [n for n, _ in fs.aggregate(per_window, **kw)]


def test_bns_hits_count_toward_the_ipc_section_they_replaced():
    # BNS 103 replaced IPC 302
    assert ranked([[(BNS, "103")]]) == ["302"]


def test_unweighted_acts_are_ignored_and_do_not_consume_depth():
    window = [("Indian Contract Act", "10"), ("Some Other Act", "5"), (IPC, "420")]
    assert ranked([window], depth=1) == ["420"]


def test_depth_limits_how_far_down_each_window_is_read():
    window = [(IPC, "302"), (IPC, "34"), (IPC, "120B")]
    assert ranked([window], depth=2) == ["302", "34"]


def test_evidence_must_recur_across_windows_to_outrank_a_single_top_hit():
    windows = [[(IPC, "302"), (IPC, "34")], [(IPC, "323"), (IPC, "34")], [(IPC, "506"), (IPC, "34")]]
    assert ranked(windows)[0] == "34"


def test_a_number_counts_once_per_window_across_acts():
    both = [(IPC, "302"), (BNS, "103")]
    once = [(IPC, "302")]
    assert dict(fs.aggregate([both]))["302"] == dict(fs.aggregate([once]))["302"]


def test_crpc_counts_but_far_less_than_ipc():
    scores = dict(fs.aggregate([[(CRPC, "482")], [(IPC, "323")]]))
    assert 0 < scores["482"] < scores["323"]


def test_ties_are_broken_deterministically():
    assert ranked([[(IPC, "34")], [(IPC, "302")]]) == ["302", "34"]
