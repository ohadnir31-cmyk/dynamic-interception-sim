from __future__ import annotations

from typing import Iterable

# The original implementation and old datasets use "NI" as the internal
# column/heuristic key. The proposal now uses "NT" (Nearest Target) because the
# rule chooses by current geometric distance, not projected intercept time.
INTERNAL_TO_DISPLAY = {
    "NI": "NT",
}

DISPLAY_TO_INTERNAL = {
    "NT": "NI",
}


def display_heuristic_name(name: object) -> str:
    """Return the proposal-facing display name for a heuristic key."""
    text = str(name)
    return INTERNAL_TO_DISPLAY.get(text, text)


def internal_heuristic_name(name: object) -> str:
    """Return the backwards-compatible internal key for a display name."""
    text = str(name)
    return DISPLAY_TO_INTERNAL.get(text, text)


def display_policy_name(policy: object) -> str:
    """Display policy labels such as 'Always NI' as 'Always NT'."""
    text = str(policy)
    for internal, display in INTERNAL_TO_DISPLAY.items():
        text = text.replace(f"Always {internal}", f"Always {display}")
        text = text.replace(f"{internal} vs", f"{display} vs")
        text = text.replace(f"vs {internal}", f"vs {display}")
    return text


def display_heuristic_list(names: Iterable[object]) -> list[str]:
    return [display_heuristic_name(name) for name in names]
