# PNW DEUCE MUSIC AI NOTEBOOK

This notebook companion documents the production implementation in `pnw_deuce_music_ai.py`.

## Pipeline

`request -> strict Pydantic normalization -> immutable plans -> deterministic theory/drum/bass/melody engines -> validated Type-1 MIDI`

## Setup

```python
!pip install -q "pydantic>=2.0,<3.0" numpy torch pytest
```

The production MIDI writer uses only the Python standard library. `mido` and `pretty_midi` are optional tools for downstream inspection, not required by the serializer.

## Cells 00–05

The implementation provides runtime detection, headroom normalization, immutable schemas, strict key/mode/time-signature/instrument validation, and `compile_song_plan()`.

## Cells 06–09

`build_theory_plan`, `build_drum_plan`, `build_bass_plan`, and `build_melody_plan` produce frozen Pydantic models. Generation is seeded with `random.Random(seed)`. No process-randomized Python hash is used.

## Cell 10 — MIDI export

The exporter writes a Type-1 SMF at 480 PPQ with a conductor track and channel tracks for drums (9), 808 (0), and melody (1). Time signatures are converted to ticks per bar instead of assuming 4/4. Input notes are rejected when they are out of range, overlap on a channel/pitch, or exceed the song boundary.

## Cell 11 — rendering and security

Audio buffers are normalized to a configured peak ceiling. MIDI output accepts only safe relative paths and rejects traversal or absolute paths.

## Tests

Run:

```bash
python -m pytest -q tests
```
