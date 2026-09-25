import pytest
import mido
from pydantic import ValidationError

import pnw_deuce_music_ai as pipeline


def test_schema_is_immutable():
    plan = pipeline.build_song_plan(101)
    with pytest.raises(ValidationError):
        plan.tempo_bpm = 140


def test_mido_type_one_480_ppq_and_tracks():
    mid = pipeline.export_mido_midi(pipeline.build_song_plan(101))
    assert isinstance(mid, mido.MidiFile)
    assert mid.type == 1
    assert mid.ticks_per_beat == 480
    assert [track[0].name for track in mid.tracks] == ["track_name"]
    assert len(mid.tracks) == 4


def test_invalid_midi_input_is_rejected():
    with pytest.raises(TypeError):
        pipeline.export_mido_midi(object())
