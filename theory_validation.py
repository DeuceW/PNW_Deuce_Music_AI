from __future__ import annotations

from typing import Any


def get_theory_event_for_bar(theory_plan: Any, global_bar: int) -> Any:
    """Return the validated theory event for ``global_bar``.

    Supports the repository's ``TheorySection.events`` shape and the
    bar-oriented ``section.bars`` shape used by downstream plans. Malformed
    sections fail with descriptive ValueError exceptions instead of leaking
    IndexError or AttributeError failures.
    """
    if theory_plan is None or not hasattr(theory_plan, "sections"):
        raise ValueError("TheoryPlan is invalid or missing 'sections'.")

    all_events: list[Any] = []
    for section in theory_plan.sections:
        section_name = getattr(section, "section_name", getattr(section, "name", "unknown"))
        if hasattr(section, "events"):
            items = section.events
        elif hasattr(section, "bars"):
            items = section.bars
        else:
            raise ValueError(f"Theory section '{section_name}' is missing events or bars.")
        if not items:
            raise ValueError(f"Theory section '{section_name}' contains no valid bars.")
        for item in items:
            if not hasattr(item, "global_bar"):
                raise ValueError("Theory bar object is missing required attribute 'global_bar'.")
            all_events.append(item)

    if not all_events:
        raise ValueError("TheoryPlan contains zero total events across all sections.")

    matching_event = next((item for item in all_events if item.global_bar == global_bar), None)
    if matching_event is None:
        available = [item.global_bar for item in all_events]
        raise ValueError(
            f"TheoryPlan validation failed: Missing theory event for global bar {global_bar}. "
            f"Available global bars: {available}"
        )
    return matching_event
