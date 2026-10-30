#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce zstd native-library chunks and store their JAR entries without ZIP compression."""

import argparse
import concurrent.futures
import shutil
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

CHUNK_SIZE = 32 * 1024 * 1024
MINIMUM_SIZE = 256 * 1024 * 1024


def _compress(source, output, offset, length, level):
    crc = 0
    with source.open("rb") as input_file, output.open("wb") as output_file:
        input_file.seek(offset)
        process = subprocess.Popen(
            ["zstd", "--quiet", "--single-thread", "-%d" % level, "--stdout"],
            stdin=subprocess.PIPE,
            stdout=output_file,
        )
        try:
            while length:
                data = input_file.read(min(1024 * 1024, length))
                if not data:
                    raise IOError(
                        "Native library changed during chunking: %s" % source
                    )
                process.stdin.write(data)
                crc = zlib.crc32(data, crc)
                length -= len(data)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(
                    "zstd failed for %s at offset %d" % (source, offset)
                )
        except BaseException:
            process.kill()
            process.wait()
            raise
    return crc & 0xFFFFFFFF


def produce(
    source,
    minimum_size=MINIMUM_SIZE,
    chunk_size=CHUNK_SIZE,
    level=9,
    workers=12,
):
    source = Path(source)
    if not source.is_file() or source.stat().st_size < minimum_size:
        return False
    source_stat = source.stat()
    size = source_stat.st_size
    chunks = Path(str(source) + ".chunks")
    manifest = Path(str(source) + ".chunks.properties")
    if chunks.exists() or manifest.exists():
        raise RuntimeError("Chunk output already exists for %s" % source)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=source.name + ".chunks.tmp-", dir=str(source.parent)
        )
    )
    temporary_manifest = Path(str(temporary) + ".properties")
    count = (size + chunk_size - 1) // chunk_size
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = [
                pool.submit(
                    _compress,
                    source,
                    temporary / ("%05d" % i),
                    i * chunk_size,
                    min(chunk_size, size - i * chunk_size),
                    level,
                )
                for i in range(count)
            ]
            checksums = [future.result() for future in futures]
        after = source.stat()
        if (
            after.st_size != size
            or after.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise IOError(
                "Native library changed during chunking: %s" % source
            )
        lines = [
            "format.version=1",
            "library.size=%d" % size,
            "chunk.size=%d" % chunk_size,
            "chunk.count=%d" % count,
            "chunk.compression=zstd",
        ]
        lines.extend(
            "chunk.%05d.crc32=%08x" % (i, crc)
            for i, crc in enumerate(checksums)
        )
        temporary_manifest.write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )
        temporary.rename(chunks)
        temporary_manifest.rename(manifest)
        source.unlink()
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if temporary_manifest.exists():
            temporary_manifest.unlink()
        shutil.rmtree(chunks, ignore_errors=True)
        if manifest.exists():
            manifest.unlink()
        raise
    return True


def store(root, archive):
    root, archive = Path(root), Path(archive)
    entries = []
    for manifest in root.rglob("*.chunks.properties"):
        properties = dict(
            line.split("=", 1)
            for line in manifest.read_text(encoding="ascii").splitlines()
            if "=" in line
        )
        if properties.get("chunk.compression") != "zstd":
            continue
        library = Path(str(manifest)[: -len(".chunks.properties")])
        for i in range(int(properties["chunk.count"])):
            chunk = Path(str(library) + ".chunks") / ("%05d" % i)
            if not chunk.is_file():
                raise IOError("Missing native chunk %s" % chunk)
            entries.append(str(chunk.relative_to(root)))
    if not entries:
        return 0
    subprocess.run(
        ["jar", "u0f", str(archive.resolve()), *entries],
        cwd=str(root),
        check=True,
    )
    with zipfile.ZipFile(archive) as jar:
        for name in entries:
            if jar.getinfo(name).compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("Native chunk is not STORED: %s" % name)
    return len(entries)


def verify(archive):
    checked = 0
    with zipfile.ZipFile(archive) as jar:
        names = set(jar.namelist())
        for manifest_name in sorted(
            name for name in names if name.endswith(".chunks.properties")
        ):
            properties = dict(
                line.split("=", 1)
                for line in jar.read(manifest_name)
                .decode("ascii")
                .splitlines()
                if "=" in line
            )
            if properties.get("chunk.compression") != "zstd":
                continue
            if properties.get("format.version") != "1":
                raise RuntimeError(
                    "Unsupported chunk format: %s" % manifest_name
                )
            library_name = manifest_name[: -len(".chunks.properties")]
            if library_name in names:
                raise RuntimeError(
                    "Conventional library beside %s" % manifest_name
                )
            size = int(properties["library.size"])
            chunk_size = int(properties["chunk.size"])
            count = int(properties["chunk.count"])
            if (
                size <= 0
                or chunk_size <= 0
                or count != (size + chunk_size - 1) // chunk_size
            ):
                raise RuntimeError("Invalid chunk layout: %s" % manifest_name)
            for index in range(count):
                name = library_name + ".chunks/%05d" % index
                entry = jar.getinfo(name)
                if entry.compress_type != zipfile.ZIP_STORED:
                    raise RuntimeError("Native chunk is not STORED: %s" % name)
                result = subprocess.run(
                    ["zstd", "-q", "-d", "--stdout"],
                    input=jar.read(entry),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                expected_size = min(chunk_size, size - index * chunk_size)
                expected_crc = int(properties["chunk.%05d.crc32" % index], 16)
                if (
                    len(result.stdout) != expected_size
                    or zlib.crc32(result.stdout) & 0xFFFFFFFF != expected_crc
                ):
                    raise RuntimeError(
                        "Native chunk size or CRC mismatch: %s" % name
                    )
                checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    pack = sub.add_parser("produce")
    pack.add_argument("library")
    pack.add_argument("--minimum-size", type=int, default=MINIMUM_SIZE)
    pack.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    pack.add_argument("--level", type=int, default=9)
    pack.add_argument("--workers", type=int, default=12)
    pack.add_argument("--require-loader", type=Path)
    jar = sub.add_parser("store")
    jar.add_argument("root")
    jar.add_argument("archive")
    check = sub.add_parser("verify")
    check.add_argument("archive")
    args = parser.parse_args()
    if args.command is None:
        parser.error("A command is required")
    if args.command == "produce":
        if args.minimum_size < 0 or args.chunk_size <= 0 or args.workers <= 0:
            parser.error("Invalid chunk size or worker count")
        if args.require_loader is not None and (
            not args.require_loader.is_file()
            or "CHUNK_COMPRESSION_KEY" not in args.require_loader.read_text()
        ):
            parser.error(
                "cuDF submodule must contain the zstd NativeDepsLoader"
            )
        sys.stdout.write(
            "Chunked native library: %s"
            % produce(
                args.library,
                args.minimum_size,
                args.chunk_size,
                args.level,
                args.workers,
            )
            + "\n"
        )
    elif args.command == "store":
        sys.stdout.write(
            "Stored %d zstd native chunks\n" % store(args.root, args.archive)
        )
    else:
        sys.stdout.write(
            "Verified %d zstd native chunks\n" % verify(args.archive)
        )


if __name__ == "__main__":
    main()
