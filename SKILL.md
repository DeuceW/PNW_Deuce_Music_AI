# PNW Deuce Music AI — Authoritative V1.1.1 Skill

This file is the repository-level authority for the V1.1.1 baseline. Any implementation, refactor, exporter, or test change must preserve all invariants below.

## Eight Authoritative Invariants

1. **Canonical implementation**  
   `pnw_deuce_music_ai.py` is the canonical implementation and the source of truth for the V1.1.1 pipeline.

2. **Strict immutable schemas**  
   All plan/schema models use Pydantic v2 with `frozen=True` and `extra="forbid"`. Plans are immutable and unknown fields are rejected.

3. **Deterministic planning**  
   `build_song_plan(seed)` uses explicit seeded RNG and stable integer pitch-class mappings. Python `hash()` must never participate in generated musical state.

4. **Single canonical MIDI path**  
   `export_mido_midi(plan)` is the sole canonical MIDI exporter. Custom standard-library MIDI serializers are deprecated and must not become an alternate production path.

5. **Canonical MIDI format**  
   MIDI output is Type 1 at exactly 480 PPQ.

6. **Canonical track contract**  
   The exporter produces exactly four tracks, in this order: `Conductor`, `Drums`, `808`, `Melody`. Drums use MIDI channel 10 (zero-based 9), 808 uses channel 1 (zero-based 0), and Melody uses channel 2 (zero-based 1).

7. **Explicit audio/headroom behavior**  
   MIDI export performs no silent normalization, limiting, clipping repair, or hidden humanization. Audio/headroom behavior must remain explicit and deterministic.

8. **Independent verification gate**  
   V1.1.1 is not locked or independently verified unless the repository test suite completes successfully with `pytest -v`. Determinism verification must cover both serialized `SongPlan` output and canonical mido MIDI bytes across subprocesses and `PYTHONHASHSEED` variations.

## Verification Gate

Before declaring the V1.1.1 baseline locked:

```bash
pytest -v
```

A failing assertion, collection error, dependency error, or implementation error means the baseline is not independently verified.

The expected MIDI track assertion is:

```python
assert [track[0].name for track in mid.tracks] == [
    "Conductor",
    "Drums",
    "808",
    "Melody",
]
```
