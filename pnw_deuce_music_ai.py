from __future__ import annotations

import json
import random
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

PITCH_CLASS = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
REGISTERED_INSTRUMENTS = ("drums", "808", "piano", "electric_guitar", "strings", "synth", "brass", "bass")
VALID_MODES = frozenset({"major", "minor"})
SCALE_INTERVALS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
STEPS_PER_BAR = 16
PPQ = 480


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def runtime_check() -> Dict[str, Any]:
    if torch is not None and torch.cuda.is_available():
        return {"cuda_available": True, "device": torch.cuda.get_device_name(0)}
    return {"cuda_available": False, "device": "cpu"}


def verify_headroom(buffer: np.ndarray, peak_limit: float = 0.99) -> np.ndarray:
    if not isinstance(buffer, np.ndarray):
        raise TypeError("buffer must be a numpy.ndarray")
    if not 0 < peak_limit <= 1:
        raise ValueError("peak_limit must be in (0, 1]")
    if not buffer.size:
        return buffer.copy()
    peak = float(np.max(np.abs(buffer)))
    return buffer.copy() if peak <= peak_limit else buffer * (peak_limit / peak)


class ProjectMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str = Field(..., min_length=1)
    artist: str = "PNW Deuce"
    bpm: int = Field(..., ge=60, le=180)
    key: str
    mode: str = Field(..., pattern=r"^(minor|major)$")
    time_signature: str = "4/4"
    seed: int = 20260918


class SongRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str = Field(..., min_length=1)
    artist: str = "PNW Deuce"
    bpm: int = Field(..., ge=60, le=180)
    key: str
    mode: str = Field(..., pattern=r"^(minor|major)$")
    time_signature: str = "4/4"
    seed: int = 20260918
    total_bars: int = Field(8, ge=1, le=256)
    instruments: Tuple[str, ...] = ("drums", "808", "synth", "bass")

    @field_validator("key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in PITCH_CLASS:
            raise ValueError(f"Unsupported key: {value}")
        return value

    @field_validator("time_signature")
    @classmethod
    def valid_time_signature(cls, value: str) -> str:
        if not re.fullmatch(r"\d+/(?:1|2|4|8|16|32|64)", value):
            raise ValueError(f"Invalid time signature: {value}")
        numerator, denominator = map(int, value.split("/"))
        if numerator < 1 or numerator > 255:
            raise ValueError("time-signature numerator must be 1..255")
        return value

    @field_validator("instruments")
    @classmethod
    def valid_instruments(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        values = tuple(str(value).strip() for value in values)
        invalid = sorted(set(values) - set(REGISTERED_INSTRUMENTS))
        if invalid:
            raise ValueError(f"Unregistered instruments: {invalid}")
        return values


def compile_song_plan(raw_request: Mapping[str, Any]) -> SongRequest:
    if not isinstance(raw_request, Mapping):
        raise TypeError("raw_request must be a mapping")
    return SongRequest(**raw_request)


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


def _section_ranges(total_bars: int) -> List[Tuple[int, int]]:
    return [(start, min(total_bars, start + total_bars // 4 - 1)) for start in range(1, total_bars + 1, max(1, total_bars // 4))][:4]


def build_theory_plan(song: SongRequest) -> TheoryPlan:
    rng = random.Random(song.seed)
    scale_names = ("C", "D", "E", "F", "G", "A", "B")
    key_letter = song.key[0]
    key_index = scale_names.index(key_letter)
    qualities = ("min", "dim", "maj", "min", "min", "maj", "maj") if song.mode == "minor" else ("maj", "min", "min", "maj", "maj", "min", "dim")
    intervals = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6)}
    sections = []
    for index, (start, end) in enumerate(_section_ranges(song.total_bars), 1):
        events = []
        for bar in range(start, end + 1):
            root = scale_names[(key_index + bar - 1) % 7]
            quality = qualities[(bar + rng.randint(0, 2)) % 7]
            root_pc = PITCH_CLASS[root]
            bass_note = 36 + ((bar + root_pc) % 12)  # stable across Python processes
            chord = TheoryChord(root_name=root, quality=quality, pitch_classes=tuple((root_pc + i) % 12 for i in intervals[quality]))
            events.append(TheoryEvent(global_bar=bar, section_bar=bar - start + 1, chord=chord, voicing=TheoryVoicing(midi_notes=(bass_note, bass_note + 4, bass_note + 7), bass_note=bass_note)))
        sections.append(TheorySection(section_name=f"section_{index}", start_bar=start, end_bar=end, events=tuple(events)))
    return TheoryPlan(title=song.title, artist=song.artist, key=song.key, mode=song.mode, bpm=song.bpm, time_signature=song.time_signature, seed=song.seed, sections=tuple(sections))


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
    sections = []
    for index, (start, end) in enumerate(_section_ranges(song.total_bars), 1):
        bars = []
        for bar in range(start, end + 1):
            hits = []
            for step in range(16):
                for instrument, steps, base in (("kick", (0, 4, 8, 12), 95), ("snare", (4, 12), 90), ("closed_hat", (1, 3, 5, 7, 9, 11, 13, 15), 65)):
                    if step in steps:
                        hits.append(DrumHit(instrument=instrument, bar=bar, step=step, velocity=base + rng.randint(0, min(20, 127 - base)), event_type="normal"))
            bars.append(DrumBarEvent(global_bar=bar, section_bar=bar - start + 1, hits=tuple(hits)))
        sections.append(DrumSection(section_name=f"drums_section_{index}", start_bar=start, end_bar=end, bars=tuple(bars)))
    return DrumPlan(title=song.title, bpm=song.bpm, time_signature=song.time_signature, seed=song.seed, sections=tuple(sections))


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
    sections = []
    for section in theory.sections:
        bars = []
        for event in section.events:
            root = clamp(36 + PITCH_CLASS[event.chord.root_name], 24, 60)
            bars.append(BassBar(global_bar=event.global_bar, events=tuple(BassEvent(global_bar=event.global_bar, step=step, midi_note=root, duration_steps=2, velocity=82) for step in (0, 4, 8, 12))))
        sections.append(BassSection(section_name=section.section_name, start_bar=section.start_bar, end_bar=section.end_bar, bars=tuple(bars)))
    return BassPlan(title=song.title, bpm=song.bpm, seed=song.seed, sections=tuple(sections))


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
    sections = []
    for section in theory.sections:
        bars = []
        for event in section.events:
            root = PITCH_CLASS[event.chord.root_name]
            notes = tuple(MelodyNote(global_bar=event.global_bar, step=step, midi_note=clamp(60 + root + (step // 2) % 5 - 2, 48, 96), duration_steps=2, velocity=72) for step in range(0, 16, 2))
            bars.append(MelodyBar(global_bar=event.global_bar, notes=notes))
        sections.append(MelodySection(section_name=section.section_name, start_bar=section.start_bar, end_bar=section.end_bar, bars=tuple(bars)))
    return MelodyPlan(title=song.title, seed=song.seed, sections=tuple(sections))


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
    if not 1 <= numerator <= 255 or denominator not in (1, 2, 4, 8, 16, 32, 64):
        raise MidiValidationError("invalid time signature")
    total = ppq * 4 * numerator
    if total % denominator:
        raise MidiValidationError("time signature does not divide evenly")
    return total // denominator


def validate_midi_notes(title: str, bpm: int, time_sig: str, total_bars: int, notes: Sequence[MidiNote]) -> int:
    if not title.strip() or not 20 <= bpm <= 400 or not isinstance(total_bars, int) or total_bars < 1:
        raise MidiValidationError("invalid MIDI metadata")
    numerator, denominator = map(int, time_sig.split("/"))
    end = ticks_per_bar(numerator, denominator) * total_bars
    spans: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for index, note in enumerate(notes):
        values = (note.channel, note.pitch, note.velocity, note.start_tick, note.duration_ticks)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise MidiValidationError(f"note {index}: fields must be integers")
        if not 0 <= note.channel <= 15 or not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127 or note.start_tick < 0 or note.duration_ticks < 1 or note.start_tick + note.duration_ticks > end:
            raise MidiValidationError(f"note {index}: out of bounds")
        spans.setdefault((note.channel, note.pitch), []).append((note.start_tick, note.start_tick + note.duration_ticks))
    for key, ranges in spans.items():
        ranges.sort()
        if any(second[0] < first[1] for first, second in zip(ranges, ranges[1:])):
            raise MidiValidationError(f"overlapping notes: {key}")
    return end


def _vlq(value: int) -> bytes:
    if not 0 <= value <= 0x0FFFFFFF:
        raise MidiValidationError("VLQ out of range")
    output = [value & 0x7F]
    value >>= 7
    while value:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(output))


def _meta(delta: int, kind: int, data: bytes) -> bytes:
    return _vlq(delta) + bytes((0xFF, kind)) + _vlq(len(data)) + data


def _track(data: bytes) -> bytes:
    return b"MTrk" + struct.pack(">I", len(data)) + data


def build_midi_bytes(title: str, bpm: int, time_signature: str, total_bars: int, notes: Sequence[MidiNote]) -> bytes:
    end = validate_midi_notes(title, bpm, time_signature, total_bars, notes)
    numerator, denominator = map(int, time_signature.split("/"))
    tempo = round(60_000_000 / bpm)
    conductor = _meta(0, 3, title.encode()) + _meta(0, 0x58, bytes((numerator, denominator.bit_length() - 1, 24, 8))) + _meta(0, 0x51, tempo.to_bytes(3, "big")) + _meta(end, 0x2F, b"")
    by_channel: Dict[int, List[MidiNote]] = {}
    for note in notes:
        by_channel.setdefault(note.channel, []).append(note)
    chunks = [_track(conductor)]
    for channel in sorted(by_channel):
        events = []
        for note in by_channel[channel]:
            events.extend(((note.start_tick, 1, note.pitch, note.velocity), (note.start_tick + note.duration_ticks, 0, note.pitch, 0)))
        events.sort(key=lambda event: (event[0], event[1], event[2], event[3]))
        data = _meta(0, 3, {0: "808", 1: "melody", 9: "drums"}.get(channel, f"ch{channel}").encode())
        previous = 0
        for tick, on, pitch, velocity in events:
            data += _vlq(tick - previous) + bytes(((0x90 if on else 0x80) | channel, pitch, velocity))
            previous = tick
        data += _meta(end - previous, 0x2F, b"")
        chunks.append(_track(data))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), PPQ)
    return header + b"".join(chunks)


def export_to_midi(song_plan: SongRequest, theory_plan: TheoryPlan, drum_plan: DrumPlan, bass_plan: BassPlan, melody_plan: MelodyPlan, output_filename: str = "pnw_deuce_output.mid") -> str:
    if not output_filename or Path(output_filename).is_absolute() or ".." in Path(output_filename).parts:
        raise ValueError("output_filename must be a safe relative path")
    numerator, denominator = map(int, song_plan.time_signature.split("/"))
    step = ticks_per_bar(numerator, denominator) // STEPS_PER_BAR
    notes: List[MidiNote] = []
    drum_map = {"kick": 36, "snare": 38, "closed_hat": 42, "open_hat": 46, "crash": 49, "tom": 45}
    for section in drum_plan.sections:
        for bar in section.bars:
            for hit in bar.hits:
                notes.append(MidiNote(9, drum_map[hit.instrument], hit.velocity, (bar.global_bar - 1) * step * 16 + hit.step * step, max(60, step)))
    for section in bass_plan.sections:
        for bar in section.bars:
            for event in bar.events:
                notes.append(MidiNote(0, event.midi_note, event.velocity, (bar.global_bar - 1) * step * 16 + event.step * step, event.duration_steps * step))
    for section in melody_plan.sections:
        for bar in section.bars:
            for note in bar.notes:
                notes.append(MidiNote(1, note.midi_note, note.velocity, (bar.global_bar - 1) * step * 16 + note.step * step, note.duration_steps * step))
    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_midi_bytes(song_plan.title, song_plan.bpm, song_plan.time_signature, song_plan.total_bars, notes))
    return str(path)


def render_safe_audio_stems(total_samples: int = 48000, sample_rate: int = 44100) -> np.ndarray:
    if total_samples <= 0 or sample_rate <= 0:
        raise ValueError("sample count and sample rate must be positive")
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    return verify_headroom(np.column_stack((np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 440 * t))).astype(np.float32))


def demo() -> None:
    song = compile_song_plan({"title": "Midnight Signal", "bpm": 92, "key": "C", "mode": "minor", "total_bars": 8})
    theory = build_theory_plan(song)
    export_to_midi(song, theory, build_drum_plan(song), build_bass_plan(song, theory), build_melody_plan(song, theory))
    print(json.dumps(runtime_check(), indent=2))


if __name__ == "__main__":
    demo()
