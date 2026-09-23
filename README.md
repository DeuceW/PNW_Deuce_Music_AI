# PNW Deuce Music AI

A deterministic, compiler-style music planning system. It validates structured requests, creates immutable Pydantic plans, and writes Type-1 Standard MIDI files with a dependency-free serializer.

## Features
- Pydantic v2 models with `frozen=True` and `extra="forbid"`
- Deterministic theory, drum, bass, and melody generation
- Cross-process stable seeded output; no use of Python's randomized `hash()`
- Custom standard-library MIDI serializer (`PPQ = 480`), not a mido-based writer
- MIDI bounds, overlap, timing, and metadata validation before writing
- Headroom-safe NumPy audio utility and safe relative output paths

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pnw_deuce_music_ai.py
```

## Example

```python
from pnw_deuce_music_ai import (
    compile_song_plan, build_theory_plan, build_drum_plan,
    build_bass_plan, build_melody_plan, export_to_midi,
)

song = compile_song_plan({
    "title": "Midnight Signal", "bpm": 92, "key": "C", "mode": "minor",
    "time_signature": "4/4", "seed": 20260918, "total_bars": 8,
})
theory = build_theory_plan(song)
output = export_to_midi(
    song, theory, build_drum_plan(song), build_bass_plan(song, theory),
    build_melody_plan(song, theory), "midnight_signal.mid",
)
print(output)
```

## Architecture

`request -> strict normalization -> immutable plans -> deterministic engines -> validated MIDI`

Invalid data is rejected rather than silently repaired. The repository's tests cover imports, schema immutability, MIDI validation, and cross-process determinism.
