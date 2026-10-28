#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Build the measured cmix-views compressor; never run the enwik9 benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys


MODEL = "weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4"
CC = "clang++-18 -DEXGF_C233_X4=1 -DWR_AFD_CONTROL_OFF=0 -DEXGF_ELIDE_LEGACY_COUPLED=1"
EXPECTED = {
    "english.hints": (205067, "7c16cf40569ad5907585ae79fff48ffdf4b534c9286a3e9f9d54f29f2ac4613c"),
    "new_article_order": (115870, "c9f8b1001bfbc041a087eab65e5caa609eb87bb8ecf297cfcdbcdaffa7a1dd19"),
    MODEL: (3649198, "141ee13c316811157bea167846b3f45472affff7adaa1b8b263e30621e490885"),
    "cmix.cutguard": (None, "9400e4d63c8db70964908393870d44e3c50889cddbaf03ae853816c85dfce711"),
    "cmix.pak": (196120, "85635096cabbd3f9d8fd6e0030912e23c82d0c71067ebf3a86f640d7c1e3d433"),
    "cmix.cut.dec": (180895, "10bf08ee13535ce849f0e05f149b305ae7053307f67f9981578f30e3c868181a"),
    "hints.bin": (57493, "611fa86022e77f8a1cafcdc42bdc73a9f0fa81a1d4c27298de0b75d498e9acbc"),
    "order.bin": (43778, "b0d47bd2c97ce57271f682b5bd1af25f2aabba4baa75d6c9215f476d82ddebca"),
    "comp9": (3946605, "d9f7277fed3a945a8cce1f80b48c17c600ff5023d3ca70469485537476827955"),
}
PACK_CHECKS = (
    "cut_dpath: clean - no .encoder_only function on the -d path.",
    "CUT-Gegenprobe voll: bitidentisch",
    "CUT-Gegenprobe dec: Praefix bitidentisch",
    "CUT-Stubprobe voll: bitidentisch",
    "CUT-Stubprobe dec: Praefix bitidentisch",
)
TOOLS = (
    "make", "clang++-18", "clang-18", "ld.lld-18", "llvm-strip-18",
    "llvm-objcopy-18", "gcc", "objcopy", "objdump", "readelf", "nm",
    "strip", "gawk", "awk", "bash", "python3", "cmp", "stat", "head",
    "cp", "chmod", "mkdir", "rm", "dirname", "basename",
)


class BuildError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BuildError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, name):
    require(path.is_file() and not path.is_symlink(), f"Missing or linked input/output: {path}")
    size, expected_hash = EXPECTED[name]
    require(size is None or path.stat().st_size == size, f"Unexpected size: {path}")
    require(sha256(path) == expected_hash, f"SHA-256 mismatch: {path}")


def source_fingerprint(source):
    require(source.is_dir() and not source.is_symlink(), "codec/ must be a real source directory.")
    paths = sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix())
    require(not any(p.is_symlink() for p in paths), "Source links are not supported; include real files.")
    files = [p for p in paths if p.is_file()]
    for name in ("makefile", "encoder_split.lds", "runtime_exports.map",
                 "tools_src/packer/build_packer.sh", "tools_src/packer/cut_dpath.py", MODEL):
        require((source / name).is_file(), f"Missing source input: {name}")
    # Canonical tree fingerprint: UTF-8 'sha256  POSIX-relative-path\n',
    # sorted by relative path. No separate manifest or private receipt is read.
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{sha256(path)}  {path.relative_to(source).as_posix()}\n".encode("utf-8"))
    return digest.hexdigest()


def stop_owned_group(process):
    # Every launched command has a new session; never signal another job.
    # Signal before reaping the leader, keeping its PID reserved against reuse.
    try:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def terminate(signum, frame):
    raise BuildError(f"Received signal {signum}")


def run(label, command, cwd, logs, env, timeout=1800):
    print(f"{label}...", flush=True)
    log = logs / f"{label}.log"
    with log.open("xb") as output:
        process = subprocess.Popen(
            [str(part) for part in command], cwd=cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_owned_group(process)
            raise
    require(code == 0, f"{label} exited {code}; see {log}")
    return log


def final_records(log):
    records = {}
    for line in log.read_text(encoding="utf-8", errors="strict").splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s+(\{.*\})", line)
        if not match:
            continue
        tag, payload = match.groups()
        value = json.loads(payload)
        require(isinstance(value, dict), f"Invalid diagnostic in {log}")
        if tag in ("PAGE_COUNTER", "PREPARED_PROGRESS"):
            continue
        require(tag not in records, f"Duplicate final diagnostic {tag} in {log}")
        records[tag] = line
    require(bool(records), f"Missing final diagnostics in {log}")
    return records


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=package / "build-work",
                        help="new build/output directory; parent must exist")
    args = parser.parse_args()
    require(sys.platform == "linux" and platform.machine() == "x86_64", "Use Linux x86-64.")
    require(os.geteuid() != 0, "Run as an ordinary user, not root.")
    signal.signal(signal.SIGTERM, terminate)
    require(not args.work_dir.is_symlink(), "The work directory must not be a link.")
    work = args.work_dir.resolve()
    require(not work.exists(), f"Refusing existing work/output directory: {work}")
    require(work.parent.is_dir(), "The work directory's parent must already exist.")
    require(re.fullmatch(r"/[A-Za-z0-9_./-]+", str(work)) is not None,
            "Use a work path containing only letters, digits, /, _, . and - (no spaces).")
    source, inputs = package / "codec", package / "build-inputs"
    raw_inputs = {
        "english.hints": inputs / "english.hints",
        "new_article_order": source / "src/readalike_prepr/data/new_article_order",
    }
    require(source.resolve() not in work.parents and inputs.resolve() not in work.parents,
            "The work directory must not be inside codec/ or build-inputs/.")
    require(inputs.is_dir() and not inputs.is_symlink(), "build-inputs/ must be a real directory.")
    original_source = source_fingerprint(source)
    check_file(source / MODEL, MODEL)
    for name, path in raw_inputs.items():
        check_file(path, name)
    for name in TOOLS:
        require(shutil.which(name) is not None, f"Missing prerequisite: {name}; install separately.")
    # Do not inherit make/compiler flags, shell startup scripts or loader overrides.
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "LANG": "C"}
    work.mkdir()  # Exclusive: no reuse, overwrite or automatic cleanup.
    logs = work / "logs"
    logs.mkdir()
    run("check-awk", ["awk", 'BEGIN { exit(strtonum("0x10") != 16) }'], work, logs, env, 10)
    core, auxiliary = work / "core", work / "auxiliary"
    shutil.copytree(source, core)
    shutil.copytree(source, auxiliary)
    require(source_fingerprint(core) == original_source, "Full-build source copy differs.")
    require(source_fingerprint(auxiliary) == original_source, "AUX source copy differs.")
    build = ["make", "-j1", f"CC={CC}", "STRIP_FLAG=", "prof_use"]
    run("build-core", build, core, logs, env)
    shutil.copyfile(core / "cmix", core / "cmix.cutguard")
    check_file(core / "cmix.cutguard", "cmix.cutguard")
    run("strip-core", ["llvm-strip-18", "--strip-all", core / "cmix"], core, logs, env, 60)
    run("trim-core", ["objcopy", "--remove-section=.comment",
                      "--remove-section=.note.gnu.property", "--remove-section=.note.gnu.build-id",
                      "--remove-section=.note.ABI-tag", core / "cmix"], core, logs, env, 60)
    packed = core / "pak_pack"
    pack_env = dict(env, CC="clang-18", OBJCOPY_SS="llvm-objcopy-18")
    pack_log = run("pack-core", ["bash", core / "tools_src/packer/build_packer.sh",
                                 core / "cmix", packed], core, logs, pack_env, 600)
    pack_text = pack_log.read_text(encoding="utf-8")
    require(all(marker in pack_text for marker in PACK_CHECKS), "Packing guard/probe evidence is incomplete.")
    for name in ("cmix.pak", "cmix.cut.dec"):
        check_file(packed / name, name)
    aux_build = list(build)
    aux_build[2] += " -DAUX_BUILD=1"
    run("build-auxiliary", aux_build, auxiliary, logs, env)
    blobs = []
    for raw_name, blob_name in (("english.hints", "hints.bin"), ("new_article_order", "order.bin")):
        encode_dir, decode_dir = work / f"encode-{raw_name}", work / f"decode-{raw_name}"
        encode_dir.mkdir()
        decode_dir.mkdir()
        blob, restored = encode_dir / blob_name, decode_dir / "restored.raw"
        encode_log = run(f"encode-{raw_name}", [auxiliary / "cmix", "-c", raw_inputs[raw_name], blob],
                         encode_dir, logs, env, 300)
        check_file(blob, blob_name)
        with blob.open("rb") as stream:
            require(stream.read(1)[0] & 0x40, f"Missing AUX format flag: {blob_name}")
        decode_log = run(f"decode-{raw_name}", [packed / "cmix.cut.dec", "-D", blob, restored],
                         decode_dir, logs, env, 300)
        check_file(restored, raw_name)
        require(restored.read_bytes() == raw_inputs[raw_name].read_bytes(), f"Roundtrip differs: {raw_name}")
        require(final_records(encode_log) == final_records(decode_log), f"Diagnostics differ: {raw_name}")
        blobs.append(blob)
    require(source_fingerprint(source) == original_source, "The distributed source changed during the build.")
    for name, path in raw_inputs.items():
        check_file(path, name)
    parts = [packed / "cmix.pak", *blobs, core / MODEL]
    values = (blobs[0].stat().st_size, blobs[1].stat().st_size, 0, (core / MODEL).stat().st_size)
    require(values == (57493, 43778, 0, 3649198), "Unexpected container component lengths.")
    output = work / "comp9"
    with output.open("xb") as stream:
        for part in parts:
            with part.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
        stream.write(struct.pack("<4i", *values))
    check_file(output, "comp9")
    output.chmod(0o755)
    print(f"PASS: {output}\n{EXPECTED['comp9'][0]} bytes\nSHA-256 {EXPECTED['comp9'][1]}")
    print("Construction-input inverses passed. No enwik9 encode/decode was run. Logs remain in the work directory.")


if __name__ == "__main__":
    try:
        main()
    except (BuildError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Build stopped: {error}\nExisting inputs and build outputs were not removed.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Build interrupted; outputs were retained.", file=sys.stderr)
        sys.exit(130)
