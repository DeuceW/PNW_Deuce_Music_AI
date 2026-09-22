from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# ============================================================
# PNW DEUCE MUSIC AI — CORE CONSTANTS
# ============================================================

PITCH_CLASS: Dict[str, int] = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

REGISTERED_INSTRUMENTS = (
    "drums",
    "808",
    "piano",
    "electric_guitar",
    "strings",
    "synth",
    "brass",
    "bass",
)
VALID_MODES = frozenset({"major", "minor"})
SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}

MIDI_C4 = 60
STEPS_PER_BAR = 16
PPQ = 480

# ============================================================
# CELL 00 — SETUP / IMPORTS
# ============================================================

print("PNW DEUCE MUSIC AI V1.1.1: Setup Complete.")

# ============================================================
# CELL 01 — CORE CONFIGURATION
# ============================================================


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_pitch_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Pitch must be a string")
    token = name.strip().upper()
    mapping = {
        "C": "C",
        "C#": "C#",
        "DB": "DB",
        "D": "D",
        "D#": "D#",
        "EB": "EB",
        "E": "E",
        "F": "F",
        "F#": "F#",
        "GB": "GB",
        "G": "G",
        "G#": "G#",
        "AB": "AB",
        "A": "A",
        "A#": "A#",
        "BB": "BB",
        "B": "B",
    }
    if token not in mapping:
        raise ValueError(f"Invalid pitch class: {name}")
    return mapping[token] if token in mapping else token


# ============================================================
# CELL 02 — GPU / RUNTIME CHECK
# ============================================================


def runtime_check() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": False, "device": "cpu"}
    if torch is not None and torch.cuda.is_available():
        info["cuda_available"] = True
        info["device"] = torch.cuda.get_device_name(0)
    return info


# ============================================================
# CELL 03 — AUDIO UTILS / NORMALIZATION
# ============================================================


def verify_headroom(buffer: np.ndarray, peak_limit: float = 0.99) -> np.ndarray:
    if not isinstance(buffer, np.ndarray):
        raise TypeError("buffer must be a numpy.ndarray")
    if buffer.size == 0:
        return buffer.copy()
    max_val = float(np.max(np.abs(buffer)))
    if max_val > peak_limit:
        return buffer * (peak_limit / max_val)
    return buffer.copy()


# ============================================================
# CELL 04 — IMMUTABLE MODELS (V1.1.1)
# ============================================================


class ProjectMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str = Field(..., min_length=1)
    artist: str = "PNW Deuce"
    bpm: int = Field(..., ge=60, le=180)
    key: str
    mode: str = Field(..., pattern=r"^(minor|major)$")
    time_signature: str = "4/4"
    seed: int = Field(default_factory=lambda: 20260918)


class SongRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str = Field(..., min_length=1)
    artist: str = "PNW Deuce"
    bpm: int = Field(..., ge=60, le=180)
    key: str = Field(..., min_length=1)
    mode: str = Field(..., pattern=r"^(minor|major)$")
    time_signature: str = "4/4"
    seed: int = Field(default_factory=lambda: 20260918)
    total_bars: int = Field(default=8, ge=1, le=256)
    instruments: Tuple[str, ...] = ("drums", "808", "synth", "bass")

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip()
        if value not in PITCH_CLASS and value.upper() not in PITCH_CLASS:
            raise ValueError(f"Unsupported key: {value}")
        return value.upper() if value.upper() in PITCH_CLASS else value

    @field_validator("time_signature")
    @classmethod
    def validate_time_signature(cls, value: str) -> str:
        if not re.fullmatch(r"\d+/\d+", value):
            raise ValueError(f"Invalid time signature: {value}")
        num, den = map(int, value.split("/"))
        if num <= 0 or den <= 0:
            raise ValueError(f"Time signature must be positive: {value}")
        if den & (den - 1):
            raise ValueError(f"Denominator must be a power of two: {value}")
        return value

    @field_validator("instruments")
    @classmethod
    def validate_instruments(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        cleaned = tuple(str(item).strip() for item in value)
        invalid = [item for item in cleaned if item not in REGISTERED_INSTRUMENTS]
        if invalid:
            raise ValueError(f"Unregistered instruments: {invalid}")
        return cleaned


# ============================================================
# SECTION 05 — SONG DIRECTOR / INPUT NORMALIZER
# ============================================================


def compile_song_plan(raw_request: Mapping[str, Any]) -> SongRequest:
    if not isinstance(raw_request, Mapping):
        raise TypeError("raw_request must be a mapping")
    return SongRequest(**raw_request)


# ============================================================
# CELL 06 — MUSIC THEORY ENGINE
# ============================================================


class TheoryChord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root_name: str
    quality: str
    pitch_classes: Tuple[int, ...]


class TheoryVoicing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    midi_notes: Tuple[int, ...]
    bass_note: int


class TheoryEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    section_bar: int = Field(..., ge=1)
    chord: TheoryChord
    voicing: TheoryVoicing


class TheorySection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    section_name: str
    start_bar: int
    end_bar: int
    events: Tuple[TheoryEvent, ...]


class TheoryPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str
    artist: str
    key: str
    mode: str
    bpm: int
    time_signature: str
    seed: int
    sections: Tuple[TheorySection, ...]

    def export_json(self) -> str:
        return self.model_dump_json(indent=4)


def _next_chord_quality(mode: str, index: int) -> str:
    progression = ["maj", "min", "min", "maj", "maj", "min", "dim"] if mode == "major" else ["min", "dim", "maj", "min", "min", "maj", "maj"]
    return progression[index % len(progression)]


def _root_from_key(key: str, mode: str, offset: int = 0) -> str:
    notes = ["C", "D", "E", "F", "G", "A", "B"]
    key_root = key.upper()
    key_index = notes.index(key_root) if key_root in notes else 0
    return notes[(key_index + offset) % len(notes)]


def _compute_pitch_classes(root: str, quality: str) -> Tuple[int, ...]:
    root_pc = PITCH_CLASS[root]
    intervals = {
        "maj": (0, 4, 7),
        "min": (0, 3, 7),
        "dim": (0, 3, 6),
    }
    pc = tuple((root_pc + interval) % 12 for interval in intervals[quality])
    return pc


def build_theory_plan(song: SongRequest) -> TheoryPlan:
    rng = random.Random(song.seed)
    sections: List[TheorySection] = []
    for section_index in range(1, 5):
        start_bar = (section_index - 1) * max(1, song.total_bars // 4) + 1
        end_bar = min(song.total_bars, section_index * max(1, song.total_bars // 4))
        events: List[TheoryEvent] = []
        for bar in range(start_bar, end_bar + 1):
            quality = _next_chord_quality(song.mode, (bar + rng.randint(0, 2)) % 7)
            root = _root_from_key(song.key, song.mode, (bar - 1) % 7)
            pitch_classes = _compute_pitch_classes(root, quality)
            bass_note = 36 + ((bar + root.__hash__()) % 12)
            voicing = TheoryVoicing(
                midi_notes=(bass_note, bass_note + 4, bass_note + 7),
                bass_note=bass_note,
            )
            chord = TheoryChord(root_name=root, quality=quality, pitch_classes=pitch_classes)
            events.append(
                TheoryEvent(
                    global_bar=bar,
                    section_bar=bar - start_bar + 1,
                    chord=chord,
                    voicing=voicing,
                )
            )
        sections.append(
            TheorySection(
                section_name=f"section_{section_index}",
                start_bar=start_bar,
                end_bar=end_bar,
                events=tuple(events),
            )
        )

    return TheoryPlan(
        title=song.title,
        artist=song.artist,
        key=song.key,
        mode=song.mode,
        bpm=song.bpm,
        time_signature=song.time_signature,
        seed=song.seed,
        sections=tuple(sections),
    )


# ============================================================
# CELL 07 — DRUM ENGINE
# ============================================================


class DrumHit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    instrument: str = Field(..., pattern=r"^(kick|snare|closed_hat|open_hat|crash|tom)$")
    bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=16)
    velocity: int = Field(..., ge=1, le=127)
    event_type: str = Field(..., pattern=r"^(normal|accent|fill)$")


class DrumBarEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    section_bar: int = Field(..., ge=1)
    hits: Tuple[DrumHit, ...]


class DrumSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    section_name: str
    start_bar: int
    end_bar: int
    bars: Tuple[DrumBarEvent, ...]


class DrumPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str
    bpm: int
    time_signature: str
    seed: int
    sections: Tuple[DrumSection, ...]

    def export_json(self) -> str:
        return self.model_dump_json(indent=4)


def build_drum_plan(song: SongRequest) -> DrumPlan:
    rng = random.Random(song.seed)
    sections: List[DrumSection] = []
    bars_per_section = max(1, song.total_bars // 4)
    for section_index in range(1, 5):
        start_bar = (section_index - 1) * bars_per_section + 1
        end_bar = min(song.total_bars, section_index * bars_per_section)
        bar_events: List[DrumBarEvent] = []
        for bar in range(start_bar, end_bar + 1):
            hits: List[DrumHit] = []
            kick_steps = {0, 4, 8, 12}
            snare_steps = {4, 12}
            hat_steps = {1, 3, 5, 7, 9, 11, 13, 15}
            for step in range(STEPS_PER_BAR):
                if step in kick_steps:
                    hits.append(
                        DrumHit(
                            instrument="kick",
                            bar=bar,
                            step=step,
                            velocity=90 + rng.randint(0, 20),
                            event_type="normal",
                        )
                    )
                if step in snare_steps:
                    hits.append(
                        DrumHit(
                            instrument="snare",
                            bar=bar,
                            step=step,
                            velocity=80 + rng.randint(0, 20),
                            event_type="normal",
                        )
                    )
                if step in hat_steps:
                    hits.append(
                        DrumHit(
                            instrument="closed_hat",
                            bar=bar,
                            step=step,
                            velocity=55 + rng.randint(0, 25),
                            event_type="normal",
                        )
                    )
            bar_events.append(
                DrumBarEvent(
                    global_bar=bar,
                    section_bar=bar - start_bar + 1,
                    hits=tuple(hits),
                )
            )
        sections.append(
            DrumSection(
                section_name=f"drums_section_{section_index}",
                start_bar=start_bar,
                end_bar=end_bar,
                bars=tuple(bar_events),
            )
        )
    return DrumPlan(
        title=song.title,
        bpm=song.bpm,
        time_signature=song.time_signature,
        seed=song.seed,
        sections=tuple(sections),
    )


# ============================================================
# CELL 08 — 808 / BASS ENGINE
# ============================================================


class BassEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=16)
    midi_note: int = Field(..., ge=24, le=60)
    duration_steps: int = Field(..., ge=1, le=16)
    velocity: int = Field(..., ge=1, le=127)


class BassBar(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    events: Tuple[BassEvent, ...]


class BassSection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    section_name: str
    start_bar: int
    end_bar: int
    bars: Tuple[BassBar, ...]


class BassPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str
    bpm: int
    seed: int
    sections: Tuple[BassSection, ...]

    def export_json(self) -> str:
        return self.model_dump_json(indent=4)


def build_bass_plan(song: SongRequest, theory: TheoryPlan) -> BassPlan:
    sections: List[BassSection] = []
    for theory_section in theory.sections:
        bars: List[BassBar] = []
        for bar in range(theory_section.start_bar, theory_section.end_bar + 1):
            events: List[BassEvent] = []
            chord = theory_section.events[bar - theory_section.start_bar].chord
            root_pc = PITCH_CLASS.get(chord.root_name, 0)
            midi_root = 36 + (root_pc % 12)
            for step in (0, 4, 8, 12):
                events.append(
                    BassEvent(
                        global_bar=bar,
                        step=step,
                        midi_note=clamp(midi_root, 24, 60),
                        duration_steps=2,
                        velocity=82,
                    )
                )
            bars.append(BassBar(global_bar=bar, events=tuple(events)))
        sections.append(
            BassSection(
                section_name=theory_section.section_name,
                start_bar=theory_section.start_bar,
                end_bar=theory_section.end_bar,
                bars=tuple(bars),
            )
        )
    return BassPlan(title=song.title, bpm=song.bpm, seed=song.seed, sections=tuple(sections))


# ============================================================
# CELL 09 — MELODY / MOTIF ENGINE
# ============================================================


class MelodyNote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=16)
    midi_note: int = Field(..., ge=48, le=96)
    duration_steps: int = Field(..., ge=1, le=16)
    velocity: int = Field(..., ge=1, le=127)


class MelodyBar(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    notes: Tuple[MelodyNote, ...]


class MelodySection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    section_name: str
    start_bar: int
    end_bar: int
    bars: Tuple[MelodyBar, ...]


class MelodyPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str
    seed: int
    sections: Tuple[MelodySection, ...]

    def export_json(self) -> str:
        return self.model_dump_json(indent=4)


def build_melody_plan(song: SongRequest, theory: TheoryPlan) -> MelodyPlan:
    sections: List[MelodySection] = []
    for theory_section in theory.sections:
        melody_bars: List[MelodyBar] = []
        for bar in range(theory_section.start_bar, theory_section.end_bar + 1):
            section_event = theory_section.events[bar - theory_section.start_bar]
            chord = section_event.chord
            root_pc = PITCH_CLASS.get(chord.root_name, 0)
            melody_notes: List[MelodyNote] = []
            for step in range(0, 16, 2):
                pitch = 60 + root_pc + ((step // 2) % 5) - 2
                pitch = clamp(pitch, 48, 96)
                melody_notes.append(
                    MelodyNote(
                        global_bar=bar,
                        step=step,
                        midi_note=pitch,
                        duration_steps=2,
                        velocity=72,
                    )
                )
            melody_bars.append(MelodyBar(global_bar=bar, notes=tuple(melody_notes)))
        sections.append(
            MelodySection(
                section_name=theory_section.section_name,
                start_bar=theory_section.start_bar,
                end_bar=theory_section.end_bar,
                bars=tuple(melody_bars),
            )
        )
    return MelodyPlan(title=song.title, seed=song.seed, sections=tuple(sections))


# ============================================================
# CELL 10 — MIDI / RENDER EXPORTER
# ============================================================


@dataclass(frozen=True)
class MidiNote:
    channel: int
    pitch: int
    velocity: int
    start_tick: int
    duration_ticks: int


class MidiValidationError(ValueError):
    pass


def ticks_per_bar(numerator: int, denominator: int, ppq: int = PPQ) -> int:
    if numerator < 1 or numerator > 255:
        raise MidiValidationError(f"time signature numerator out of range: {numerator}")
    if denominator not in (1, 2, 4, 8, 16, 32, 64):
        raise MidiValidationError(f"time signature denominator must be a power of two: {denominator}")
    total = ppq * 4 * numerator
    if total % denominator:
        raise MidiValidationError(f"{numerator}/{denominator} does not divide evenly at PPQ {ppq}")
    return total // denominator


def validate_midi_notes(title: str, bpm: int, time_sig: str, total_bars: int, notes: Sequence[MidiNote]) -> int:
    if not isinstance(title, str) or not title.strip():
        raise MidiValidationError("title must be a non-empty string")
    if not (20 <= bpm <= 400):
        raise MidiValidationError(f"bpm out of range 20-400: {bpm}")
    if not isinstance(total_bars, int) or total_bars < 1:
        raise MidiValidationError("total_bars must be an int >= 1")

    numerator, denominator = map(int, time_sig.split("/"))
    tpb = ticks_per_bar(numerator, denominator)
    end_of_song = tpb * total_bars

    for index, note in enumerate(notes):
        for field_name in ("channel", "pitch", "velocity", "start_tick", "duration_ticks"):
            value = getattr(note, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise MidiValidationError(f"note {index}: {field_name} must be an int")
        if not 0 <= note.channel <= 15:
            raise MidiValidationError(f"note {index}: channel out of range 0-15: {note.channel}")
        if not 0 <= note.pitch <= 127:
            raise MidiValidationError(f"note {index}: pitch out of range 0-127: {note.pitch}")
        if not 1 <= note.velocity <= 127:
            raise MidiValidationError(f"note {index}: velocity out of range 1-127: {note.velocity}")
        if note.start_tick < 0:
            raise MidiValidationError(f"note {index}: negative start_tick: {note.start_tick}")
        if note.duration_ticks < 1:
            raise MidiValidationError(f"note {index}: duration_ticks must be >= 1: {note.duration_ticks}")
        if note.start_tick + note.duration_ticks > end_of_song:
            raise MidiValidationError(
                f"note {index}: ends at tick {note.start_tick + note.duration_ticks}, past song end {end_of_song}"
            )

    by_channel_pitch: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for note in notes:
        key = (note.channel, note.pitch)
        by_channel_pitch.setdefault(key, []).append((note.start_tick, note.start_tick + note.duration_ticks))

    for (channel, pitch), spans in by_channel_pitch.items():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise MidiValidationError(f"overlapping notes on channel {channel}, pitch {pitch}: tick {s2} starts before {e1}")

    return end_of_song


def _vlq(value: int) -> bytes:
    if not 0 <= value <= 0x0FFFFFFF:
        raise MidiValidationError(f"VLQ out of range: {value}")
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _meta(delta: int, kind: int, data: bytes) -> bytes:
    return _vlq(delta) + bytes([0xFF, kind]) + _vlq(len(data)) + data


def _track_chunk(events: bytes) -> bytes:
    return b"MTrk" + len(events).to_bytes(4, byteorder="big", signed=False) + events


def _build_conductor_track(title: str, bpm: int, time_signature: str, end_tick: int) -> bytes:
    numerator, denominator = map(int, time_signature.split("/"))
    tempo = round(60_000_000 / bpm)
    denominator_log2 = denominator.bit_length() - 1
    data = b""
    data += _meta(0, 0x03, title.encode("utf-8"))
    data += _meta(0, 0x58, bytes([numerator, denominator_log2, 24, 8]))
    data += _meta(0, 0x51, bytes([tempo >> 16, (tempo >> 8) & 0xFF, tempo & 0xFF]))
    data += _meta(end_tick, 0x2F, b"")
    return _track_chunk(data)


def _build_note_track(name: str, channel: int, notes: Sequence[MidiNote], end_tick: int) -> bytes:
    events: List[Tuple[int, int, int, int]] = []
    for note in notes:
        events.append((note.start_tick, 1, note.pitch, note.velocity))
        events.append((note.start_tick + note.duration_ticks, 0, note.pitch, 0))
    events.sort(key=lambda item: (item[0], item[1]))

    data = _meta(0, 0x03, name.encode("utf-8"))
    last_tick = 0
    for tick, onoff, pitch, velocity in events:
        status = (0x90 if onoff == 1 else 0x80) | channel
        data += _vlq(tick - last_tick) + bytes([status, pitch, velocity])
        last_tick = tick
    data += _meta(end_tick - last_tick, 0x2F, b"")
    return _track_chunk(data)


def build_midi_bytes(title: str, bpm: int, time_signature: str, total_bars: int, notes: Sequence[MidiNote]) -> bytes:
    end_tick = validate_midi_notes(title, bpm, time_signature, total_bars, notes)
    by_channel: Dict[int, List[MidiNote]] = {}
    for note in notes:
        by_channel.setdefault(note.channel, []).append(note)

    header = b"MThd" + (6).to_bytes(4, byteorder="big") + (1).to_bytes(2, byteorder="big") + (len(by_channel) + 1).to_bytes(2, byteorder="big") + PPQ.to_bytes(2, byteorder="big")

    chunks = [_build_conductor_track(title, bpm, time_signature, end_tick)]
    for channel in sorted(by_channel):
        track_name = {0: "808", 1: "melody", 9: "drums"}.get(channel, f"ch{channel}")
        chunks.append(_build_note_track(track_name, channel, by_channel[channel], end_tick))
    return header + b"".join(chunks)


def export_to_midi(
    song_plan: SongRequest,
    theory_plan: TheoryPlan,
    drum_plan: DrumPlan,
    bass_plan: BassPlan,
    melody_plan: MelodyPlan,
    output_filename: str = "pnw_deuce_output.mid",
) -> str:
    """Serialize concept plans into a standard MIDI file (Type 1)."""
    if not isinstance(output_filename, str) or not output_filename.strip():
        raise ValueError("output_filename must be a non-empty string")

    midi_notes: List[MidiNote] = []
    denominator = int(song_plan.time_signature.split("/")[1])
    ticks_per_measure = ticks_per_bar(int(song_plan.time_signature.split("/")[0]), denominator)
    step_ticks = ticks_per_measure // STEPS_PER_BAR

    for section in drum_plan.sections:
        for bar in section.bars:
            bar_start = (bar.global_bar - 1) * ticks_per_measure
            for hit in bar.hits:
                note = {"kick": 36, "snare": 38, "closed_hat": 42, "open_hat": 46, "crash": 49, "tom": 45}.get(hit.instrument, 36)
                start_tick = bar_start + hit.step * step_ticks
                midi_notes.append(MidiNote(channel=9, pitch=note, velocity=hit.velocity, start_tick=start_tick, duration_ticks=max(60, step_ticks)))

    for section in bass_plan.sections:
        for bar in section.bars:
            bar_start = (bar.global_bar - 1) * ticks_per_measure
            for event in bar.events:
                start_tick = bar_start + event.step * step_ticks
                midi_notes.append(MidiNote(channel=0, pitch=event.midi_note, velocity=event.velocity, start_tick=start_tick, duration_ticks=event.duration_steps * step_ticks))

    for section in melody_plan.sections:
        for bar in section.bars:
            bar_start = (bar.global_bar - 1) * ticks_per_measure
            for note in bar.notes:
                start_tick = bar_start + note.step * step_ticks
                midi_notes.append(MidiNote(channel=1, pitch=note.midi_note, velocity=note.velocity, start_tick=start_tick, duration_ticks=note.duration_steps * step_ticks))

    payload = build_midi_bytes(song_plan.title, song_plan.bpm, song_plan.time_signature, song_plan.total_bars, midi_notes)
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return str(output_path)


# ============================================================
# CELL 11 — AUDIO RENDER / SECURITY
# ============================================================


def render_safe_audio_stems(total_samples: int = 48000, sample_rate: int = 44100) -> np.ndarray:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    stereo[:, 0] = np.sin(np.linspace(0, 2 * np.pi * 220, total_samples, endpoint=False)) * 0.2
    stereo[:, 1] = np.sin(np.linspace(0, 2 * np.pi * 220 * 2, total_samples, endpoint=False)) * 0.2
    return verify_headroom(stereo)


def safe_output_path(filename: str) -> Path:
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    candidate = Path(filename)
    if candidate.is_absolute():
        raise ValueError("Absolute paths are forbidden")
    if any(part in {"..", "/", "\\"} for part in candidate.parts):
        raise ValueError("Path traversal is forbidden")
    return candidate


# ============================================================
# DEMO / EXAMPLE
# ============================================================


def demo() -> None:
    raw_request = {
        "title": "Midnight Signal",
        "artist": "PNW Deuce",
        "bpm": 92,
        "key": "C",
        "mode": "minor",
        "time_signature": "4/4",
        "seed": 20260918,
        "total_bars": 8,
        "instruments": ("drums", "808", "synth", "bass"),
    }
    song = compile_song_plan(raw_request)
    theory = build_theory_plan(song)
    drums = build_drum_plan(song)
    bass = build_bass_plan(song, theory)
    melody = build_melody_plan(song, theory)

    output = export_to_midi(song, theory, drums, bass, melody, "midnight_signal.mid")
    print(f"Generated MIDI: {output}")
    print(json.dumps(runtime_check(), indent=2))


if __name__ == "__main__":
    demo()
