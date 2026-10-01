import pytest

from ionpack import Layout, capacity, latin_rotation, make_layout


def test_t1_make_layout():
    lay = make_layout([2, 3, 2], 36, 1)
    assert lay.blocks == ((0, 1), (3, 4, 5), (7, 8))
    assert lay.sentinels == (2, 6)
    assert lay.width == 9
    with pytest.raises(ValueError):
        make_layout([4] * 8, 36, 1)
    assert make_layout([4] * 7, 36, 1).width == 34


def test_make_layout_gap0_and_invalid():
    lay = make_layout([2, 2], 4, 0)
    assert lay.blocks == ((0, 1), (2, 3)) and lay.sentinels == () and lay.width == 4
    for args in [([], 36, 1), ([0], 36, 1), ([2], 36, -1), ([2], 0, 1), ([2.0], 36, 1)]:
        with pytest.raises(ValueError):
            make_layout(*args)


def test_layout_rejects_overlapping_blocks():
    with pytest.raises(ValueError):
        Layout(((0, 1), (1, 2)), (), 36, 0)
    with pytest.raises(ValueError):
        Layout(((0, 1), (2, 36)), (), 36, 0)


def test_t2_capacity():
    assert capacity(4, 36, 1) == 7
    assert capacity(4, 36, 0) == 9
    assert capacity(2, 36, 1) == 12
    assert capacity(6, 36, 1) == 5
    assert capacity(8, 36, 1) == 4
    assert capacity(12, 36, 1) == 2
    for args in [(0, 36, 1), (4, 36, -1), (4, 0, 1)]:
        with pytest.raises(ValueError):
            capacity(*args)


def test_capacity_matches_make_layout():
    for w in range(1, 13):
        for gap in range(3):
            k = capacity(w, 36, gap)
            make_layout([w] * k, 36, gap)
            with pytest.raises(ValueError):
                make_layout([w] * (k + 1), 36, gap)


def test_t3_latin_rotation():
    assert latin_rotation(["a", "b", "c"], 1) == ["c", "a", "b"]
    k = 7
    items = list(range(k))
    rows = [latin_rotation(items, s) for s in range(k)]
    for slot in range(k):
        assert sorted(row[slot] for row in rows) == items
    for item in items:
        assert sorted(row.index(item) for row in rows) == list(range(k))
    for s in range(k):
        for i in items:
            assert rows[s][(i + s) % k] == i
