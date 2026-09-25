from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Dict, List, Tuple

import mido
from pydantic import BaseModel, ConfigDict, Field

PPQ = 480
STEPS_PER_BAR = 16
PITCH_CLASS: Dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
ROOT_NAMES = ("C", "C#", "D", "EB", "E", "F", "F#", "G", "AB", "A", "BB", "B")


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TheoryChord(StrictFrozenModel):
    root_name: str
    quality: str
    pitch_classes: Tuple[int, ...]


class TheoryVoicing(StrictFrozenModel):
    midi_notes: Tuple[int, ...]
    bass_note: int


class TheoryEvent(StrictFrozenModel):
    global_bar: int = Field(..., ge=1)
    section_bar: int = Field(..., ge=1)
    chord: TheoryChord
    voicing: TheoryVoicing


class TheorySection(StrictFrozenModel):
    section_name: str
    start_bar: int = Field(..., ge=1)
    end_bar: int = Field(..., ge=1)
    events: Tuple[TheoryEvent, ...]


class TheoryPlan(StrictFrozenModel):
    sections: Tuple[TheorySection, ...]


class DrumHit(StrictFrozenModel):
    instrument: str
    bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=STEPS_PER_BAR)
    velocity: int = Field(..., ge=1, le=127)


class DrumBar(StrictFrozenModel):
    global_bar: int = Field(..., ge=1)
    hits: Tuple[DrumHit, ...]


class DrumPlan(StrictFrozenModel):
    bars: Tuple[DrumBar, ...]


class BassEvent(StrictFrozenModel):
    global_bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=STEPS_PER_BAR)
    midi_note: int = Field(..., ge=24, le=60)
    duration_steps: int = Field(..., ge=1, le=STEPS_PER_BAR)
    velocity: int = Field(..., ge=1, le=127)


class BassPlan(StrictFrozenModel):
    events: Tuple[BassEvent, ...]


class MelodyNote(StrictFrozenModel):
    global_bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=STEPS_PER_BAR)
    midi_note: int = Field(..., ge=48, le=96)
    duration_steps: int = Field(..., ge=1, le=STEPS_PER_BAR)
    velocity: int = Field(..., ge=1, le=127)


class MelodyPlan(StrictFrozenModel):
    notes: Tuple[MelodyNote, ...]


class SongPlan(StrictFrozenModel):
    seed: int
    tempo_bpm: int = Field(..., ge=20, le=300)
    root_note: str
    scale_type: str
    bars: int = Field(..., ge=1, le=256)
    theory: TheoryPlan
    drums: DrumPlan
    bass: BassPlan
    melody: MelodyPlan


def _chord_quality(index: int) -> str:
    return ("min", "dim", "maj", "min", "min", "maj", "maj")[index % 7]


def _make_theory(rng: random.Random, root_pc: int, bars: int) -> TheoryPlan:
    events: List[TheoryEvent] = []
    intervals = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6)}
    for bar in range(1, bars + 1):
        root_index = (root_pc + (bar - 1) * 2) % 12
        root = ROOT_NAMES[root_index]
        quality = _chord_quality(bar + rng.randint(0, 2))
        chord_pcs = tuple((PITCH_CLASS[root] + interval) % 12 for interval in intervals[quality])
        # Stable integer arithmetic only; never use Python hash().
        bass_note = 36 + ((bar + PITCH_CLASS[root]) % 12)
        events.append(TheoryEvent(
            global_bar=bar,
            section_bar=bar,
            chord=TheoryChord(root_name=root, quality=quality, pitch_classes=chord_pcs),
            voicing=TheoryVoicing(midi_notes=(bass_note, bass_note + 3, bass_note + 7), bass_note=bass_note),
        ))
    return TheoryPlan(sections=(TheorySection(section_name="main", start_bar=1, end_bar=bars, events=tuple(events)),))


def _make_drums(rng: random.Random, bars: int) -> DrumPlan:
    result: List[DrumBar] = []
    for bar in range(1, bars + 1):
        hits: List[DrumHit] = []
        for step in range(STEPS_PER_BAR):
            if step in (0, 4, 8, 12):
                hits.append(DrumHit(instrument="kick", bar=bar, step=step, velocity=95 + rng.randint(0, 20)))
            if step in (4, 12):
                hits.append(DrumHit(instrument="snare", bar=bar, step=step, velocity=88 + rng.randint(0, 20)))
            if step % 2 == 1:
                hits.append(DrumHit(instrument="closed_hat", bar=bar, step=step, velocity=55 + rng.randint(0, 25)))
        result.append(DrumBar(global_bar=bar, hits=tuple(hits)))
    return DrumPlan(bars=tuple(result))


def _make_bass(theory: TheoryPlan) -> BassPlan:
    events = []
    for theory_event in theory.sections[0].events:
        for step in (0, 4, 8, 12):
            events.append(BassEvent(global_bar=theory_event.global_bar, step=step,
                                    midi_note=theory_event.voicing.bass_note,
                                    duration_steps=2, velocity=82))
    return BassPlan(events=tuple(events))


def _make_melody(theory: TheoryPlan) -> MelodyPlan:
    notes = []
    for theory_event in theory.sections[0].events:
        root = PITCH_CLASS[theory_event.chord.root_name]
        for step in range(0, STEPS_PER_BAR, 2):
            notes.append(MelodyNote(global_bar=theory_event.global_bar, step=step,
                                    midi_note=max(48, min(96, 60 + root + (step // 2) % 5 - 2)),
                                    duration_steps=2, velocity=72))
    return MelodyPlan(notes=tuple(notes))


def build_song_plan(seed: int) -> SongPlan:
    """Build the canonical deterministic plan for a seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    root = ROOT_NAMES[rng.randrange(len(ROOT_NAMES))]
    tempo = rng.randint(80, 140)
    bars = 16
    theory = _make_theory(rng, PITCH_CLASS[root], bars)
    drums = _make_drums(rng, bars)
    bass = _make_bass(theory)
    melody = _make_melody(theory)
    return SongPlan(seed=seed, tempo_bpm=tempo, root_note=root, scale_type="minor",
                    bars=bars, theory=theory, drums=drums, bass=bass, melody=melody)


def _append_notes(track: mido.MidiTrack, events: List[Tuple[int, int, int, int]], channel: int) -> None:
    """Append absolute-tick note events as deterministic delta-timed mido messages."""
    # note-off precedes note-on at equal ticks; remaining fields make ties stable.
    ordered = sorted(events, key=lambda event: (event[0], event[1], event[2], event[3]))
    previous = 0
    for tick, kind, pitch, velocity in ordered:
        delta = tick - previous
        message_type = "note_on" if kind else "note_off"
        track.append(mido.Message(message_type, channel=channel, note=pitch,
                                  velocity=velocity if kind else 0, time=delta))
        previous = tick


def export_mido_midi(plan: SongPlan) -> mido.MidiFile:
    """Canonical Type-1 exporter: 480 PPQ, conductor/drums/808/melody tracks."""
    if not isinstance(plan, SongPlan):
        raise TypeError("plan must be a SongPlan")
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    ticks_per_bar = PPQ * 4  # canonical plan is 4/4
    ticks_per_step = ticks_per_bar // STEPS_PER_BAR

    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Conductor", time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(plan.tempo_bpm), time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=plan.bars * ticks_per_bar))
    mid.tracks.append(conductor)

    drum_track = mido.MidiTrack()
    drum_track.append(mido.MetaMessage("track_name", name="Drums", time=0))
    drum_map = {"kick": 36, "snare": 38, "closed_hat": 42}
    drum_events = []
    for bar in plan.drums.bars:
        for hit in bar.hits:
            tick = (bar.global_bar - 1) * ticks_per_bar + hit.step * ticks_per_step
            drum_events.extend(((tick, 1, drum_map[hit.instrument], hit.velocity),
                                (tick + max(1, ticks_per_step // 2), 0, drum_map[hit.instrument], 0)))
    _append_notes(drum_track, drum_events, 9)
    drum_track.append(mido.MetaMessage("end_of_track", time=plan.bars * ticks_per_bar - sum(message.time for message in drum_track[1:])))
    mid.tracks.append(drum_track)

    for name, channel, events in (
        ("808", 0, [( (event.global_bar - 1) * ticks_per_bar + event.step * ticks_per_step, 1, event.midi_note, event.velocity) for event in plan.bass.events for _ in (0,)]),
        ("Melody", 1, [( (note.global_bar - 1) * ticks_per_bar + note.step * ticks_per_step, 1, note.midi_note, note.velocity) for note in plan.melody.notes for _ in (0,)]),
    ):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        expanded = []
        source = plan.bass.events if channel == 0 else plan.melody.notes
        for item in source:
            start = (item.global_bar - 1) * ticks_per_bar + item.step * ticks_per_step
            pitch = item.midi_note
            expanded.extend(((start, 1, pitch, item.velocity),
                             (start + item.duration_steps * ticks_per_step, 0, pitch, 0)))
        _append_notes(track, expanded, channel)
        elapsed = sum(message.time for message in track[1:])
        track.append(mido.MetaMessage("end_of_track", time=max(0, plan.bars * ticks_per_bar - elapsed)))
        mid.tracks.append(track)
    return mid


def export_to_midi(plan: SongPlan, output_path: str = "pnw_deuce_output.mid") -> str:
    """Compatibility wrapper delegating exclusively to export_mido_midi."""
    if not isinstance(output_path, str) or not output_path or Path(output_path).is_absolute() or ".." in Path(output_path).parts:
        raise ValueError("output_path must be a safe relative path")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_mido_midi(plan).save(filename=str(path))
    return str(path)


if __name__ == "__main__":
    export_to_midi(build_song_plan(20260918))
