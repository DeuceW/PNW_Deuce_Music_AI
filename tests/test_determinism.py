import json
import os
import subprocess
import sys


def _run(seed: int, hash_seed: str) -> tuple[bytes, bytes]:
    code = """
import io
import pnw_deuce_music_ai as p
plan = p.build_song_plan(SEED)
mid = p.export_mido_midi(plan)
buf = io.BytesIO()
mid.save(file=buf)
print(plan.model_dump_json())
print(buf.getvalue().hex())
""".replace("SEED", str(seed))
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, check=True)
    lines = result.stdout.splitlines()
    return lines[-2], bytes.fromhex(lines[-1].decode())


def test_cross_process_byte_determinism():
    outputs = [_run(42069, value) for value in ("0", "12345", "random")]
    assert outputs[0] == outputs[1] == outputs[2]
