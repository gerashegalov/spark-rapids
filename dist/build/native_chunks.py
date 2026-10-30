#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Produce zstd native-library chunks and store their JAR entries without ZIP compression."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
import zlib

try:
    import Queue as queue
except ImportError:
    import queue

CHUNK_SIZE = 32 * 1024 * 1024
MINIMUM_SIZE = 256 * 1024 * 1024


def _file_state(path):
    stat = os.stat(path)
    modified = getattr(stat, "st_mtime_ns", None)
    if modified is None:
        from java.nio.file import Files, Paths
        from java.util.concurrent import TimeUnit
        modified = Files.getLastModifiedTime(Paths.get(path)).to(TimeUnit.NANOSECONDS)
    return stat.st_size, modified


def _compress(source, output, offset, length, level):
    crc = 0
    with open(source, "rb") as input_file, open(output, "wb") as output_file:
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
                    raise IOError("Native library changed during chunking: %s" % source)
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


def _compress_all(source, temporary, size, chunk_size, level, workers):
    count = (size + chunk_size - 1) // chunk_size
    tasks = queue.Queue()
    for index in range(count):
        tasks.put(index)
    checksums = [None] * count
    errors = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                index = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                checksums[index] = _compress(
                    source,
                    os.path.join(temporary, "%05d" % index),
                    index * chunk_size,
                    min(chunk_size, size - index * chunk_size),
                    level,
                )
            except BaseException as error:
                lock.acquire()
                try:
                    errors.append(error)
                finally:
                    lock.release()
            finally:
                tasks.task_done()

    threads = [
        threading.Thread(target=worker)
        for unused in range(min(workers, count))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return checksums


def produce(
    source,
    minimum_size=MINIMUM_SIZE,
    chunk_size=CHUNK_SIZE,
    level=9,
    workers=12,
):
    source = str(source)
    chunks = source + ".chunks"
    manifest = source + ".chunks.properties"
    # Maven may copy a rebuilt library over the previous output while leaving
    # its old chunks in target/classes. Replace those chunks for this source.
    shutil.rmtree(chunks, ignore_errors=True)
    if os.path.exists(manifest):
        os.unlink(manifest)
    if not os.path.isfile(source) or os.path.getsize(source) < minimum_size:
        return False
    size, modified = _file_state(source)
    temporary = tempfile.mkdtemp(
        prefix=os.path.basename(source) + ".chunks.tmp-",
        dir=os.path.dirname(source) or ".",
    )
    temporary_manifest = temporary + ".properties"
    try:
        checksums = _compress_all(
            source, temporary, size, chunk_size, level, workers
        )
        if _file_state(source) != (size, modified):
            raise IOError("Native library changed during chunking: %s" % source)
        lines = [
            "format.version=1",
            "library.size=%d" % size,
            "chunk.size=%d" % chunk_size,
            "chunk.count=%d" % len(checksums),
            "chunk.compression=zstd",
        ]
        lines.extend(
            "chunk.%05d.crc32=%08x" % (index, crc)
            for index, crc in enumerate(checksums)
        )
        with open(temporary_manifest, "wb") as output:
            output.write(("\n".join(lines) + "\n").encode("ascii"))
        os.rename(temporary, chunks)
        os.rename(temporary_manifest, manifest)
        os.unlink(source)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if os.path.exists(temporary_manifest):
            os.unlink(temporary_manifest)
        shutil.rmtree(chunks, ignore_errors=True)
        if os.path.exists(manifest):
            os.unlink(manifest)
        raise
    return True


def _properties(contents):
    return dict(
        line.split("=", 1)
        for line in contents.decode("ascii").splitlines()
        if "=" in line
    )


def store(root, archive):
    root, archive = str(root), str(archive)
    entries = []
    for directory, unused_dirs, names in os.walk(root):
        for name in sorted(names):
            if not name.endswith(".chunks.properties"):
                continue
            manifest = os.path.join(directory, name)
            with open(manifest, "rb") as input_file:
                properties = _properties(input_file.read())
            if properties.get("chunk.compression") != "zstd":
                continue
            library = manifest[: -len(".chunks.properties")]
            for index in range(int(properties["chunk.count"])):
                chunk = os.path.join(library + ".chunks", "%05d" % index)
                if not os.path.isfile(chunk):
                    raise IOError("Missing native chunk %s" % chunk)
                entries.append(os.path.relpath(chunk, root).replace(os.sep, "/"))
    if not entries:
        return 0
    subprocess.check_call(
        ["jar", "u0f", os.path.abspath(archive)] + entries,
        cwd=root,
    )
    with zipfile.ZipFile(archive) as jar:
        for name in entries:
            if jar.getinfo(name).compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("Native chunk is not STORED: %s" % name)
    return len(entries)


def verify(archive):
    checked = 0
    with zipfile.ZipFile(str(archive)) as jar:
        names = set(jar.namelist())
        for manifest_name in sorted(
            name for name in names if name.endswith(".chunks.properties")
        ):
            properties = _properties(jar.read(manifest_name))
            if properties.get("chunk.compression") != "zstd":
                continue
            if properties.get("format.version") != "1":
                raise RuntimeError("Unsupported chunk format: %s" % manifest_name)
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
                process = subprocess.Popen(
                    ["zstd", "-q", "-d", "--stdout"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                output, error = process.communicate(jar.read(entry))
                if process.returncode:
                    raise subprocess.CalledProcessError(
                        process.returncode, "zstd", error
                    )
                expected_size = min(chunk_size, size - index * chunk_size)
                expected_crc = int(properties["chunk.%05d.crc32" % index], 16)
                if (
                    len(output) != expected_size
                    or zlib.crc32(output) & 0xFFFFFFFF != expected_crc
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
    pack.add_argument("--require-loader")
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
        if args.require_loader is not None:
            if not os.path.isfile(args.require_loader):
                parser.error("cuDF submodule must contain the zstd NativeDepsLoader")
            with open(args.require_loader) as loader:
                if "CHUNK_COMPRESSION_KEY" not in loader.read():
                    parser.error("cuDF submodule must contain the zstd NativeDepsLoader")
        sys.stdout.write(
            "Chunked native library: %s\n"
            % produce(
                args.library,
                args.minimum_size,
                args.chunk_size,
                args.level,
                args.workers,
            )
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
