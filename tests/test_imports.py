import pnw_deuce_music_ai as pipeline


def test_canonical_imports():
    assert hasattr(pipeline, "PITCH_CLASS")
    assert hasattr(pipeline, "SongPlan")
    assert hasattr(pipeline, "build_song_plan")
    assert hasattr(pipeline, "export_mido_midi")
    assert hasattr(pipeline, "export_to_midi")
