# PNW Deuce Music AI

The canonical implementation is `pnw_deuce_music_ai.py`. It is a deterministic, schema-first music compiler with exactly one MIDI path: **mido**.

## Contract

- Pydantic v2 models are immutable (`frozen=True`) and reject unknown fields (`extra="forbid"`).
- `build_song_plan(seed)` uses explicit seeded RNG and stable integer pitch-class mappings. Python `hash()` is never used.
- `export_mido_midi(plan)` is the sole canonical exporter.
- MIDI is Type 1 at 480 PPQ with four tracks: Conductor, Drums (channel 10 / zero-based 9), 808 (channel 1 / zero-based 0), and Melody (channel 2 / zero-based 1).
- Plans and mido serialization are deterministic across processes and `PYTHONHASHSEED` values.
- Audio/headroom behavior must remain explicit; no silent normalization is part of MIDI export.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pnw_deuce_music_ai.py
```

## Test

```bash
pytest -v
```

The subprocess determinism test compares both `SongPlan.model_dump_json()` and the bytes produced by the canonical mido exporter.
