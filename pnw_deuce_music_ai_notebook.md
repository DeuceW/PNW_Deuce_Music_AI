# PNW DEUCE MUSIC AI NOTEBOOK

This Markdown notebook captures the project architecture and the core logic for the PNW Deuce Music AI compiler.

## Overview

The project follows a deterministic compiler pipeline:

Raw User Request -> Parser -> Normalizer -> Music Rule Engine -> Schema Validator -> Immutable Song Plan -> Theory / Drum / Bass / Melody Engines -> MIDI Exporter -> Audio Render

## Section 00: Environment Setup

```python
!pip install -q "pydantic>=2.0,<3.0" mido pretty_midi numpy torch
```

## Section 01: Core Configuration

```python
PITCH_CLASS = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
REGISTERED_INSTRUMENTS = ("drums", "808", "piano", "electric_guitar", "strings", "synth", "brass", "bass")
VALID_MODES = frozenset({"major", "minor"})
SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}
```

## Section 02: GPU / Runtime

```python
import torch

if torch.cuda.is_available():
    print(f"GPU Active: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not detected. Running on CPU fallback.")
```

## Section 03: Audio Utilities

```python
import numpy as np

def verify_headroom(buffer: np.ndarray, peak_limit: float = 0.99) -> np.ndarray:
    max_val = np.max(np.abs(buffer))
    if max_val > peak_limit:
        buffer = buffer * (peak_limit / max_val)
    return buffer
```

## Section 04: Immutable Schemas

```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

class ProjectMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    title: str = Field(..., min_length=1)
    artist: str = "PNW Deuce"
    bpm: int = Field(..., ge=60, le=180)
    key: str
    mode: str = Field(..., pattern="^(minor|major)$")
    time_signature: str = "4/4"
    seed: int = Field(default_factory=lambda: 20260918)
```

## Section 05: Song Director

```python
def compile_song_plan(raw_request: dict) -> dict:
    return raw_request
```

## Section 06: Theory Engine

```python
class TheoryChord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root_name: str
    quality: str
    pitch_classes: tuple[int, ...]
```

## Section 07: Drum Engine

```python
STEPS_PER_BAR = 16
```

## Section 08: Bass Engine

```python
class BassEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=16)
    midi_note: int = Field(..., ge=24, le=60)
    duration_steps: int = Field(..., ge=1, le=16)
    velocity: int = Field(..., ge=1, le=127)
```

## Section 09: Melody Engine

```python
class MelodyNote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    global_bar: int = Field(..., ge=1)
    step: int = Field(..., ge=0, lt=16)
    midi_note: int = Field(..., ge=48, le=96)
    duration_steps: int = Field(..., ge=1, le=16)
    velocity: int = Field(..., ge=1, le=127)
```

## Section 10: MIDI Export

```python
import mido

mid = mido.MidiFile(type=1, ticks_per_beat=480)
```

## Section 11: Audio Render + Security

```python
def render_safe_audio_stems(total_samples: int = 48000, sample_rate: int = 44100) -> np.ndarray:
    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    stereo[:, 0] = np.sin(np.linspace(0, 2 * np.pi * 220, total_samples, endpoint=False)) * 0.2
    stereo[:, 1] = np.sin(np.linspace(0, 2 * np.pi * 220 * 2, total_samples, endpoint=False)) * 0.2
    return verify_headroom(stereo)
```

## Production Implementation

The full executable version is contained in `pnw_deuce_music_ai.py`.
