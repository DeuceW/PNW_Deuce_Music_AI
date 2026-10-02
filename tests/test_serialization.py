"""Serialization round-trip regression test.

SongPlan.model_dump_json() output must re-validate into an equal plan.
This converts the already-observed, working serialization property into
an explicit regression gate: serialized SongPlan -> validated SongPlan
must equal the original. Test-only; no production-code changes.
"""

import pytest

import pnw_deuce_music_ai as p


@pytest.mark.parametrize("seed", [1, 42, 101, 20260918, 999983])
def test_song_plan_json_round_trip_is_stable(seed):
    plan = p.build_song_plan(seed)
    serialized = plan.model_dump_json()
    revived = p.SongPlan.model_validate_json(serialized)
    assert revived == plan
    assert revived.model_dump_json() == serialized
