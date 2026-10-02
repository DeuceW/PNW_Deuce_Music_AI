"""Contract-enforcement tests for the hardening pass.

Covers the four identified gaps against the real models and the real
exporter (not synthetic helpers):
  F1  BassEvent / MelodyNote reject step + duration_steps > 16
  F2  DrumHit.instrument is constrained to the exporter vocabulary
  T1  MIDI channel contract (Drums=9, 808=0, Melody=1)
  T2  Plan velocities reach MIDI bytes verbatim (no silent processing)
  T3  export_to_midi path-safety rules
"""

import mido
import pytest
from pydantic import ValidationError

import pnw_deuce_music_ai as p

PPQ = p.PPQ
STEPS_PER_BAR = p.STEPS_PER_BAR
TICKS_PER_BAR = PPQ * 4
TICKS_PER_STEP = TICKS_PER_BAR // STEPS_PER_BAR
DRUM_MAP = {"kick": 36, "snare": 38, "closed_hat": 42}


# --- F1: span bound enforced on the real models -----------------------------

def test_bass_event_span_overflow_rejected_on_real_model():
    with pytest.raises(ValidationError):
        p.BassEvent(global_bar=1, step=15, midi_note=36, duration_steps=2, velocity=80)


def test_melody_note_span_overflow_rejected_on_real_model():
    with pytest.raises(ValidationError):
        p.MelodyNote(global_bar=1, step=14, midi_note=60, duration_steps=4, velocity=80)


def test_span_boundary_exactly_16_steps_accepted():
    # step + duration_steps == 16 is the legal boundary, not an overflow.
    p.BassEvent(global_bar=1, step=14, midi_note=36, duration_steps=2, velocity=80)
    p.MelodyNote(global_bar=1, step=15, midi_note=60, duration_steps=1, velocity=80)


def test_generator_never_violates_span_bound():
    for seed in (1, 42, 101, 20260918, 999983):
        plan = p.build_song_plan(seed)
        for event in plan.bass.events:
            assert event.step + event.duration_steps <= STEPS_PER_BAR
        for note in plan.melody.notes:
            assert note.step + note.duration_steps <= STEPS_PER_BAR


# --- F2: drum instrument vocabulary enforced at the schema boundary ---------

def test_drum_hit_rejects_unknown_instrument_on_real_model():
    with pytest.raises(ValidationError):
        p.DrumHit(instrument="tom", bar=1, step=0, velocity=90)


@pytest.mark.parametrize("instrument", ["kick", "snare", "closed_hat"])
def test_drum_hit_accepts_exporter_vocabulary(instrument):
    hit = p.DrumHit(instrument=instrument, bar=1, step=0, velocity=90)
    assert hit.instrument == instrument


def test_exporter_never_sees_unknown_instrument():
    # With F2 in place, an invalid instrument fails at model construction,
    # so the exporter cannot hit a raw KeyError on the drum map.
    plan = p.build_song_plan(7)
    for bar in plan.drums.bars:
        for hit in bar.hits:
            assert hit.instrument in DRUM_MAP


# --- T1: MIDI channel contract ----------------------------------------------

def _note_messages(track):
    return [m for m in track if m.type in ("note_on", "note_off")]


def test_midi_channel_contract():
    mid = p.export_mido_midi(p.build_song_plan(101))
    assert [t[0].name for t in mid.tracks] == ["Conductor", "Drums", "808", "Melody"]
    assert _note_messages(mid.tracks[0]) == []  # conductor carries meta only
    assert {m.channel for m in _note_messages(mid.tracks[1])} == {9}
    assert {m.channel for m in _note_messages(mid.tracks[2])} == {0}
    assert {m.channel for m in _note_messages(mid.tracks[3])} == {1}


# --- T2: velocities preserved verbatim; no silent processing -----------------

def _expected_note_ons(plan):
    expected = set()
    for bar in plan.drums.bars:
        for hit in bar.hits:
            tick = (bar.global_bar - 1) * TICKS_PER_BAR + hit.step * TICKS_PER_STEP
            expected.add((tick, DRUM_MAP[hit.instrument], hit.velocity))
    for event in plan.bass.events:
        tick = (event.global_bar - 1) * TICKS_PER_BAR + event.step * TICKS_PER_STEP
        expected.add((tick, event.midi_note, event.velocity))
    for note in plan.melody.notes:
        tick = (note.global_bar - 1) * TICKS_PER_BAR + note.step * TICKS_PER_STEP
        expected.add((tick, note.midi_note, note.velocity))
    return expected


def _actual_note_ons(mid):
    actual = set()
    for track in mid.tracks[1:]:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                actual.add((tick, msg.note, msg.velocity))
            assert msg.type in ("note_on", "note_off") or isinstance(msg, mido.MetaMessage), \
                f"unexpected message type in MIDI output: {msg.type}"
    return actual


def test_velocities_preserved_verbatim_no_silent_processing():
    plan = p.build_song_plan(20260918)
    mid = p.export_mido_midi(plan)
    assert _actual_note_ons(mid) == _expected_note_ons(plan)


# --- T3: export_to_midi path safety ------------------------------------------

def test_export_to_midi_rejects_absolute_path():
    plan = p.build_song_plan(5)
    with pytest.raises(ValueError):
        p.export_to_midi(plan, "/tmp/evil.mid")


def test_export_to_midi_rejects_parent_traversal():
    plan = p.build_song_plan(5)
    with pytest.raises(ValueError):
        p.export_to_midi(plan, "../evil.mid")


def test_export_to_midi_writes_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = p.build_song_plan(5)
    out = p.export_to_midi(plan, "out.mid")
    assert out == "out.mid"
    mid = mido.MidiFile(str(tmp_path / "out.mid"))
    assert mid.type == 1 and mid.ticks_per_beat == 480
