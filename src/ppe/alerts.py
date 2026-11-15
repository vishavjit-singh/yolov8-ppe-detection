"""Alert debouncing over a video stream.

Per-frame verdicts flicker: a helmet is missed for one frame, a reflection looks like a bare head.
Paging a supervisor on every flicker trains people to ignore alerts. An alert therefore opens only when
at least ``raise_after`` of the last ``window`` frames had a violation (N-of-M), and it closes only after
``clear_after`` consecutive clean frames. Every alert becomes one event with a start, an end and a peak.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class AlertEvent:
    start_frame: int
    end_frame: int | None = None
    violating_frames: int = 0
    peak_violators: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self, fps: float | None = None) -> dict:
        d = {"start_frame": self.start_frame, "end_frame": self.end_frame, "violating_frames": self.violating_frames,
             "peak_violators": self.peak_violators, "reasons": dict(sorted(self.reasons.items()))}
        if fps:
            d["start_s"] = round(self.start_frame / fps, 2)
            d["end_s"] = round(self.end_frame / fps, 2) if self.end_frame is not None else None
        return d


class AlertDebouncer:
    def __init__(self, raise_after: int = 3, window: int = 5, clear_after: int = 5):
        if not 1 <= raise_after <= window:
            raise ValueError("need 1 <= raise_after <= window")
        self.raise_after, self.window, self.clear_after = raise_after, window, clear_after
        self.recent: deque[tuple[bool, int, dict]] = deque(maxlen=window)
        self.clean_streak = 0
        self.open: AlertEvent | None = None
        self.events: list[AlertEvent] = []

    @property
    def active(self) -> bool:
        return self.open is not None

    def update(self, frame: int, violators: int, reasons: dict[str, int] | None = None) -> str | None:
        """Feed one frame. Returns "raised", "cleared" or None."""
        bad = violators > 0
        self.recent.append((bad, violators, reasons or {}))
        self.clean_streak = 0 if bad else self.clean_streak + 1
        if self.open is None:
            flags = [r[0] for r in self.recent]
            if sum(flags) >= self.raise_after:
                first_bad = frame - len(flags) + 1 + flags.index(True)
                self.open = AlertEvent(start_frame=first_bad)
                self.events.append(self.open)
                for _, v, r in self.recent:          # the frames that triggered the alert belong to it
                    self._accumulate(v, r)
                return "raised"
            return None
        self._accumulate(violators, reasons)
        if self.clean_streak >= self.clear_after:
            self.open.end_frame = frame - self.clear_after
            self.open = None
            return "cleared"
        return None

    def _accumulate(self, violators: int, reasons: dict[str, int] | None) -> None:
        if violators:
            self.open.violating_frames += 1
            self.open.peak_violators = max(self.open.peak_violators, violators)
            for k, v in (reasons or {}).items():
                self.open.reasons[k] = self.open.reasons.get(k, 0) + v

    def close(self, last_frame: int) -> None:
        """End of stream: close an open alert at the last violating frame seen."""
        if self.open is not None:
            self.open.end_frame = last_frame - self.clean_streak
            self.open = None
