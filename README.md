# PNW Deuce Music AI

A deterministic music AI compiler for generating MIDI-based song plans from structured metadata and immutable plans.

## Features
- Strict input validation with Pydantic v2
- Immutable plan models using `frozen=True` and `extra="forbid"`
- Deterministic theory, drum, bass, and melody generation
- Type-1 MIDI export using `mido`
- Headroom-safe audio normalization utilities
- Security-minded validation and output handling

## Project layout
- `pnw_deuce_music_ai.py` — full implementation of Cells 00–11 and the export pipeline
- `pnw_deuce_music_ai_notebook.md` — notebook-style Markdown summary with the module flow
- `requirements.txt` — pinned dependencies

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pnw_deuce_music_ai.py
```

## Example

```python
from pnw_deuce_music_ai import compile_song_plan, build_theory_plan, build_drum_plan, build_bass_plan, build_melody_plan, export_to_midi

raw = {
    "title": "Midnight Signal",
    "artist": "PNW Deuce",
    "bpm": 92,
    "key": "C",
    "mode": "minor",
    "time_signature": "4/4",
    "seed": 20260918,
    "total_bars": 8,
    "instruments": ["drums", "808", "synth", "bass"],
}

song = compile_song_plan(raw)
theory = build_theory_plan(song)
drum = build_drum_plan(song)
bass = build_bass_plan(song, theory)
melody = build_melody_plan(song, theory)
output = export_to_midi(song, theory, drum, bass, melody, "midnight_signal.mid")
print(f"Saved MIDI: {output}")
```

## Notes
This project follows a compiler-style architecture rather than a black-box generator:

User request -> parser -> normalizer -> music rules -> immutable plan -> engines -> MIDI/audio export

The generation is deterministic, validated, and designed to reject invalid input instead of repairing it.
