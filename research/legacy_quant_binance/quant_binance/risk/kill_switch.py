from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KillSwitch:
    armed: bool = False
    reasons: list[str] = field(default_factory=list)

    def arm(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.armed = True

    def clear(self) -> None:
        self.armed = False
        self.reasons.clear()

    def status(self) -> dict[str, object]:
        return {"armed": self.armed, "reasons": list(self.reasons)}
