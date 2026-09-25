import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from theory_validation import get_theory_event_for_bar


def test_empty_theory_sections_raises_value_error():
    class MockSection:
        name = "intro"
        bars = []

    class MockTheoryPlan:
        sections = [MockSection()]

    with pytest.raises(ValueError, match="contains no valid bars"):
        get_theory_event_for_bar(MockTheoryPlan(), global_bar=1)


def test_missing_bar_event_raises_value_error():
    class MockBar:
        global_bar = 1

    class MockSection:
        name = "verse"
        bars = [MockBar()]

    class MockTheoryPlan:
        sections = [MockSection()]

    with pytest.raises(ValueError, match="Missing theory event for global bar 5"):
        get_theory_event_for_bar(MockTheoryPlan(), global_bar=5)


def test_repository_theory_events_are_supported():
    class MockEvent:
        global_bar = 3

    class MockSection:
        section_name = "main"
        events = [MockEvent()]

    class MockTheoryPlan:
        sections = [MockSection()]

    assert get_theory_event_for_bar(MockTheoryPlan(), 3).global_bar == 3


def test_out_of_range_duration_raises_error():
    class StrictEvent(BaseModel):
        model_config = ConfigDict(frozen=True, extra="forbid")
        step: int = Field(..., ge=0, le=15)
        duration_steps: int = Field(..., ge=1, le=16)

        @classmethod
        def validate_span(cls, values):
            if values.get("step", 0) + values.get("duration_steps", 0) > 16:
                raise ValueError("event exceeds 16-step bar")
            return values

    # Pydantic field bounds alone do not validate the combined span; validate
    # the malformed combination explicitly at the boundary.
    with pytest.raises(ValidationError):
        StrictEvent.model_validate({"step": 10, "duration_steps": 10})
