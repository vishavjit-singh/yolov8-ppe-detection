import pytest

from ppe.alerts import AlertDebouncer


def feed(deb, pattern):
    out = []
    for i, ch in enumerate(pattern):
        r = deb.update(i, 1 if ch == "x" else 0, {"helmet": 1} if ch == "x" else None)
        if r:
            out.append((i, r))
    return out


def test_single_frame_flicker_never_alerts():
    deb = AlertDebouncer(3, 5, 5)
    assert feed(deb, "..x....x...x..x....") == []
    assert deb.events == []


def test_sustained_violation_raises_once_and_clears():
    deb = AlertDebouncer(3, 5, 5)
    changes = feed(deb, "....xx.xxxxx.x..........")
    assert changes == [(7, "raised"), (18, "cleared")]
    (e,) = deb.events
    assert e.start_frame == 4                 # backdated to the first violating frame in the window
    assert e.end_frame == 13                  # last violating frame
    assert e.violating_frames == 8 and e.reasons == {"helmet": 8}   # every x in the pattern


def test_open_alert_is_closed_at_end_of_stream():
    deb = AlertDebouncer(2, 3, 5)
    feed(deb, "xxxx..")
    deb.close(5)
    assert deb.events[0].end_frame == 3


def test_invalid_configuration():
    with pytest.raises(ValueError):
        AlertDebouncer(6, 5, 1)


def test_to_dict_with_fps():
    deb = AlertDebouncer(1, 1, 1)
    feed(deb, "x.")
    assert deb.events[0].to_dict(fps=10)["start_s"] == 0.0
