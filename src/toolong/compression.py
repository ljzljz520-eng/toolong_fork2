"""Random access support for compressed log files (gzip / bzip2).

The :class:`CompressionSource` interface hides the details of decompressing a
compressed log so the rest of the application can read logical (decompressed)
offsets via a growable spool file.

Features
--------

* Checkpoints map compressed offsets to logical offsets:

  - gzip: one checkpoint per *member* (a multi member ``.gz`` is a plain
    concatenation of independent gzip members; the DEFLATE dictionary resets
    at every member).
  - bzip2: one checkpoint per *block*.  Block start magics are *bit* aligned
    (not byte aligned) and blocks are independently decodable.

* Decompression happens in bounded logical segments on a background thread,
  so the UI can browse as soon as the first segment has been materialised.

* Already decoded members / blocks are also kept in a bounded, content
  addressed cache (validated with CRC32 on every read) so that random reads
  ahead of the materialised position don't have to restart from zero.

* Newline and timestamp sidecars are persisted on disk and cryptographically
  bound to the compressed file's identity (stat fingerprint + SHA256).

* Defences against decompression bombs, corrupt trailers and cache poisoning.

* A process local registry shares a single source between multiple tabs and
  owns the lifecycle of background threads and (Windows sensitive) temporary
  files.
"""

from __future__ import annotations

import array
import bz2
import hashlib
import json
import os
import platform
import queue
import struct
import tempfile
import threading
import time
import zlib

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple


IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Tunables (environment overrides are honoured, mainly for testing)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


#: Cap for a single decompressed file (MiB).
MAX_DECOMPRESSED_MB = _env_int("TOOLONG_MAX_DECOMPRESSED_MB", 8 * 1024)
#: Maximum compression ratio for a gzip member.
GZIP_MAX_RATIO = _env_int("TOOLONG_GZIP_MAX_RATIO", 2000)
#: Maximum compression ratio for a single bzip2 block.
BZIP2_MAX_RATIO = _env_int("TOOLONG_BZIP2_MAX_RATIO", 200_000)
#: Total size of the content addressed cache on disk (MiB).
CACHE_LIMIT_MB = _env_int("TOOLONG_CACHE_MB", 256)
#: Logical size of a materialised segment notification.
SEGMENT_TARGET = _env_int("TOOLONG_SEGMENT_KB", 1024) * 1024
#: Cache units larger than this are not persisted (cheap to regenerate / huge).
MAX_CACHED_UNIT = 32 * 1024 * 1024
#: Sidecars larger than this many lines are not written.
MAX_SIDECAR_LINES = 8_000_000

READ_SIZE = 64 * 1024
FINGERPRINT_PREFIX = 64 * 1024


# ---------------------------------------------------------------------------
# Bzip2 CRC32 (MSB first polynomial, *not* the reflected zlib variant)
# ---------------------------------------------------------------------------

def _make_bzip2_crc_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return tuple(table)


_BZIP2_CRC_TABLE = _make_bzip2_crc_table()


def bzip2_crc32(data: bytes, crc: int = 0xFFFFFFFF) -> int:
    """Bzip2's CRC32 (poly 0x04C11DB7, init/final xor 0xFFFFFFFF)."""
    table = _BZIP2_CRC_TABLE
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[(crc >> 24) ^ byte]
    return crc ^ 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CompressionError(Exception):
    """Base class for compression related failures."""


class CorruptCompressionError(CompressionError):
    """The compressed data is corrupt (bad magic, CRC or trailer)."""


class DecompressionBombError(CompressionError):
    """Decompressed output exceeded the configured safety limits."""


class CachePoisonError(CompressionError):
    """A cache / sidecar entry failed integrity validation."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileIdentity:
    """Stable identity of a compressed file.

    ``fingerprint`` is cheap to compute (head + tail + stat) and good enough
    to bind a sidecar before the whole file has been read; ``sha256`` is the
    digest of the *complete* compressed stream and is filled in once the
    materialiser has read it all.
    """

    path: str
    size: int
    mtime_ns: int
    inode: int
    device: int
    fingerprint: str
    sha256: str | None = None
    format: str = ""

    @property
    def stable_id(self) -> str:
        """Short id used for cache / sidecar file names."""
        return hashlib.sha256(
            f"{self.path}|{self.device}:{self.inode}|{self.size}|{self.mtime_ns}".encode(
                "utf-8"
            )
        ).hexdigest()

    def with_sha256(self, sha256: str) -> "FileIdentity":
        object.__setattr__(self, "sha256", sha256) if False else None
        return FileIdentity(
            self.path,
            self.size,
            self.mtime_ns,
            self.inode,
            self.device,
            self.fingerprint,
            sha256,
            self.format,
        )

    def to_meta(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "inode": self.inode,
            "device": self.device,
            "fingerprint": self.fingerprint,
            "sha256": self.sha256,
            "format": self.format,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "FileIdentity":
        return cls(
            path=meta["path"],
            size=int(meta["size"]),
            mtime_ns=int(meta["mtime_ns"]),
            inode=int(meta["inode"]),
            device=int(meta["device"]),
            fingerprint=meta["fingerprint"],
            sha256=meta.get("sha256"),
            format=meta.get("format", ""),
        )

    def meta_matches(self, meta: dict[str, Any]) -> bool:
        """Whether a sidecar header matches this identity."""
        if self.sha256 and meta.get("sha256") and self.sha256 != meta["sha256"]:
            return False
        return (
            meta.get("path") == self.path
            and int(meta.get("size", -1)) == self.size
            and int(meta.get("mtime_ns", -1)) == self.mtime_ns
            and int(meta.get("inode", -1)) == self.inode
            and int(meta.get("device", -1)) == self.device
            and meta.get("fingerprint") == self.fingerprint
            and meta.get("format") == self.format
        )


def compute_identity(path: str, format: str) -> FileIdentity:
    """Build a :class:`FileIdentity` for a regular file path."""
    stat = os.stat(path)
    fingerprint = hashlib.sha256()
    fingerprint.update(struct.pack("<QQQQ", stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    with open(path, "rb", buffering=0) as raw:
        head = raw.read(FINGERPRINT_PREFIX)
        fingerprint.update(head)
        if stat.st_size > FINGERPRINT_PREFIX:
            raw.seek(max(0, stat.st_size - FINGERPRINT_PREFIX))
            fingerprint.update(raw.read(FINGERPRINT_PREFIX))
    return FileIdentity(
        path=str(Path(path).resolve()),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        inode=stat.st_ino,
        device=stat.st_dev,
        fingerprint=fingerprint.hexdigest(),
        format=format,
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class Checkpoint(NamedTuple):
    """Maps a compressed position to a logical position.

    Gzip members:
        ``start`` / ``end`` are *byte* offsets of the member in the compressed
        stream.
    Bzip2 blocks:
        ``start`` / ``end`` are *bit* offsets of the block (the block magic)
        and the following block magic / end of stream magic.
    """

    index: int
    start: int
    end: int
    logical_start: int
    logical_length: int
    crc: int
    level: int = 0

    @property
    def logical_end(self) -> int:
        return self.logical_start + self.logical_length


# ---------------------------------------------------------------------------
# Filesystem helpers (secure temp files + cross platform locking)
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    override = os.environ.get("TOOLONG_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        uid = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
        base = Path(tempfile.gettempdir()) / f"toolong-cache{uid}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except OSError:
        pass
    if not base.is_dir() or base.is_symlink():
        raise CompressionError(f"Unsafe cache directory {base!r}")
    return base


class _FileLock:
    """Best effort advisory lock, used to coordinate cache eviction."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        self._fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if IS_WINDOWS:
                import msvcrt

                msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except OSError:
            # Locking is an optimisation, never fatal.
            pass
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is None:
            return
        try:
            if IS_WINDOWS:
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(self._fd)
            self._fd = None


def _secure_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Atomically write ``data`` with restrictive permissions."""
    if path.is_symlink():
        path.unlink()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_secure(path: Path) -> bytes:
    if path.is_symlink():
        raise CachePoisonError(f"Refusing to read symlinked cache entry {path.name}")
    fd = os.open(str(path), os.O_RDONLY)
    try:
        st = os.fstat(fd)
        if (st.st_mode & 0o077) and hasattr(os, "getuid") and st.st_uid == os.getuid():
            raise CachePoisonError(f"Cache entry {path.name} is group/world accessible")
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Bounded content addressed cache
# ---------------------------------------------------------------------------

class ContentAddressedCache:
    """Disk backed, bounded, content addressed cache of decoded units.

    Entries are named after the SHA256 of (identity + compressed bytes).  On
    read every entry is re-validated (magic, identity, length, CRC32) so a
    tampered entry is detected rather than served (cache poisoning).
    """

    MAGIC = b"TLOB1\n"

    def __init__(self, root: Path | None = None, limit_mb: int = CACHE_LIMIT_MB) -> None:
        self.root = root or cache_dir()
        self.limit = limit_mb * 1024 * 1024
        self._lock = threading.Lock()

    @staticmethod
    def key(identity: FileIdentity, unit_index: int, compressed: bytes) -> str:
        h = hashlib.sha256()
        h.update(identity.stable_id.encode("ascii"))
        h.update(struct.pack("<I", unit_index))
        h.update(compressed)
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.bin"

    def get(
        self,
        key: str,
        identity: FileIdentity,
        unit_index: int,
        logical_start: int,
    ) -> bytes | None:
        path = self._path(key)
        try:
            blob = _read_secure(path)
        except FileNotFoundError:
            return None
        except (OSError, CachePoisonError):
            self._discard(path)
            return None
        try:
            if not blob.startswith(self.MAGIC):
                raise CachePoisonError("bad magic")
            header_end = blob.index(b"\n", len(self.MAGIC))
            meta = json.loads(blob[len(self.MAGIC) : header_end])
            data = blob[header_end + 1 :]
            if key != path.stem or meta.get("key") != key:
                raise CachePoisonError("key mismatch")
            if int(meta.get("unit", -1)) != unit_index:
                raise CachePoisonError("unit mismatch")
            if int(meta.get("logical_start", -1)) != logical_start:
                raise CachePoisonError("offset mismatch")
            if identity.stable_id != meta.get("identity"):
                raise CachePoisonError("identity mismatch")
            if len(data) != int(meta.get("length", -1)):
                raise CachePoisonError("length mismatch")
            if f"{zlib.crc32(data) & 0xffffffff:08x}" != meta.get("crc32"):
                raise CachePoisonError("crc32 mismatch")
            # Update atime/mtime so LRU eviction reflects real usage.
            os.utime(path, None)
            return data
        except (CachePoisonError, ValueError, KeyError, json.JSONDecodeError):
            # Poisoned / garbage entry: quarantine and regenerate.
            self._discard(path)
            return None

    def put(
        self,
        key: str,
        identity: FileIdentity,
        unit_index: int,
        logical_start: int,
        data: bytes,
    ) -> None:
        if not data or len(data) > MAX_CACHED_UNIT:
            return
        path = self._path(key)
        if path.exists():
            return
        meta = {
            "key": key,
            "identity": identity.stable_id,
            "unit": unit_index,
            "logical_start": logical_start,
            "length": len(data),
            "crc32": f"{zlib.crc32(data) & 0xffffffff:08x}",
        }
        blob = self.MAGIC + json.dumps(meta, separators=(",", ":")).encode() + b"\n" + data
        try:
            _secure_write(path, blob)
        except OSError:
            return
        self._enforce_limit()

    def _discard(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def _enforce_limit(self) -> None:
        """Delete least recently used entries until under the high watermark."""
        with self._lock, _FileLock(self.root / ".evict.lock"):
            try:
                entries = [
                    p
                    for p in self.root.glob("*.bin")
                    if not p.name.startswith((".", "sidecar-"))
                ]
            except OSError:
                return
            total = 0
            stats: list[tuple[Path, int, float]] = []
            for path in entries:
                try:
                    st = path.stat()
                except OSError:
                    continue
                total += st.st_size
                stats.append((path, st.st_size, st.st_mtime))
            high_water = int(self.limit * 0.85)
            if total <= self.limit:
                return
            stats.sort(key=lambda item: item[2])
            for path, size, _ in stats:
                if total <= high_water:
                    break
                try:
                    path.unlink()
                    total -= size
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Sidecar store (newline + timestamp indices bound to file identity)
# ---------------------------------------------------------------------------

class SidecarStore:
    """Persisted newline / timestamp indices.

    Every sidecar carries the full :class:`FileIdentity` metadata and a
    payload checksum, so a stale or tampered sidecar is never mistaken for a
    valid index.
    """

    MAGIC = b"TLSC1\n"
    _TIMESTAMP = struct.Struct("=IQd")  # line no, logical offset, epoch seconds

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or cache_dir()

    def _path(self, identity: FileIdentity, kind: str) -> Path:
        return self.root / f"sidecar-{identity.stable_id[:16]}-{kind}.bin"

    def _write(self, identity: FileIdentity, kind: str, payload: bytes, extra: dict) -> None:
        header = {
            "identity": identity.to_meta(),
            "kind": kind,
            "payload_length": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            **extra,
        }
        blob = self.MAGIC + json.dumps(header, separators=(",", ":")).encode() + b"\n" + payload
        _secure_write(self._path(identity, kind), blob)

    def _read(self, identity: FileIdentity, kind: str) -> tuple[dict, bytes] | None:
        path = self._path(identity, kind)
        try:
            blob = _read_secure(path)
        except (FileNotFoundError, OSError, CachePoisonError):
            return None
        try:
            if not blob.startswith(self.MAGIC):
                raise CachePoisonError("bad magic")
            header_end = blob.index(b"\n", len(self.MAGIC))
            header = json.loads(blob[len(self.MAGIC) : header_end])
            payload = blob[header_end + 1 :]
            stored_identity = FileIdentity.from_meta(header["identity"])
            # Everything except the (possibly not yet known) full sha must match.
            other = stored_identity.to_meta()
            if not identity.meta_matches(other):
                return None
            if len(payload) != header.get("payload_length"):
                raise CachePoisonError("sidecar length mismatch")
            if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
                raise CachePoisonError("sidecar checksum mismatch")
            return header, payload
        except (CachePoisonError, KeyError, ValueError, json.JSONDecodeError):
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def load_breaks(self, identity: FileIdentity) -> list[int] | None:
        loaded = self._read(identity, "breaks")
        if loaded is None:
            return None
        _, payload = loaded
        if len(payload) % 8:
            return None
        return list(array.array("Q", payload))

    def save_breaks(self, identity: FileIdentity, breaks: array.array) -> None:
        if len(breaks) > MAX_SIDECAR_LINES:
            return
        self._write(identity, "breaks", breaks.tobytes(), {"count": len(breaks)})

    def load_timestamps(
        self, identity: FileIdentity
    ) -> list[tuple[int, int, float]] | None:
        loaded = self._read(identity, "timestamps")
        if loaded is None:
            return None
        _, payload = loaded
        if len(payload) % self._TIMESTAMP.size:
            return None
        return [
            tuple(self._TIMESTAMP.unpack_from(payload, offset))
            for offset in range(0, len(payload), self._TIMESTAMP.size)
        ]

    def save_timestamps(
        self, identity: FileIdentity, records: list[tuple[int, int, float]]
    ) -> None:
        if len(records) > MAX_SIDECAR_LINES:
            return
        payload = b"".join(self._TIMESTAMP.pack(*record) for record in records)
        self._write(identity, "timestamps", payload, {"count": len(records)})

    def discard(self, identity: FileIdentity) -> None:
        for kind in ("breaks", "timestamps"):
            try:
                self._path(identity, kind).unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Growable spool file
# ---------------------------------------------------------------------------

class SpoolFile:
    """Append only temporary file holding materialised, mmap-able bytes.

    On POSIX the file is unlinked immediately (anonymous inode); on Windows it
    must stay named for as long as a file descriptor / mmap is open, so it is
    removed in :meth:`close` instead.
    """

    def __init__(self, prefix: str = "tl-spool-") -> None:
        directory = cache_dir()
        fd, name = tempfile.mkstemp(prefix=prefix, dir=str(directory))
        self._fd = fd
        self._path = Path(name)
        self._position = 0
        self._lock = threading.Lock()
        if not IS_WINDOWS:
            try:
                self._path.unlink()
            except OSError:
                pass

    @property
    def fileno(self) -> int:
        return self._fd

    @property
    def size(self) -> int:
        return self._position

    def append(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if IS_WINDOWS:
                os.lseek(self._fd, 0, os.SEEK_END)
            view = memoryview(data)
            while view:
                written = os.write(self._fd, view)
                view = view[written:]
            self._position += len(data)

    def pread(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        with self._lock:
            if not IS_WINDOWS:
                return os.pread(self._fd, length, offset)
            position = os.lseek(self._fd, 0, os.SEEK_CUR)
            try:
                os.lseek(self._fd, offset, os.SEEK_SET)
                chunks = []
                remaining = length
                while remaining:
                    chunk = os.read(self._fd, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks)
            finally:
                os.lseek(self._fd, position, os.SEEK_CUR)

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
        if IS_WINDOWS:
            try:
                self._path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Raw input: regular file (seekable) or arbitrary stream (tees to disk)
# ---------------------------------------------------------------------------

class _RawInput:
    """Raw compressed input with SHA256 hashing and non seekable teeing."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "rb", buffering=0)
        self.seekable = self._file.seekable()
        self.hasher = hashlib.sha256()
        self.bytes_read = 0
        self._tee: SpoolFile | None = None
        self._tee_lock = threading.Lock()
        # On Windows random reads and sequential reads share one file position;
        # on-demand reader threads must therefore exclude the materialiser.
        self._seek_lock = threading.Lock()

    @property
    def random_access(self) -> bool:
        return self.seekable or self._tee is not None

    def read(self, size: int) -> bytes:
        if IS_WINDOWS and self.seekable:
            with self._seek_lock:
                chunk = self._file.read(size)
        else:
            chunk = self._file.read(size)
        if chunk:
            self.hasher.update(chunk)
            self.bytes_read += len(chunk)
            if not self.seekable:
                if self._tee is None:
                    self._tee = SpoolFile("tl-tee-")
                self._tee.append(chunk)
        return chunk

    def pread(self, offset: int, length: int) -> bytes:
        if self.seekable:
            if not IS_WINDOWS:
                return os.pread(self._file.fileno(), length, offset)
            with self._seek_lock:
                position = os.lseek(self._file.fileno(), 0, os.SEEK_CUR)
                try:
                    os.lseek(self._file.fileno(), offset, os.SEEK_SET)
                    return self._file.read(length)
                finally:
                    os.lseek(self._file.fileno(), position, os.SEEK_SET)
        with self._tee_lock:
            assert self._tee is not None
            return self._tee.pread(offset, length)

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass
        if self._tee is not None:
            self._tee.close()
            self._tee = None


# ---------------------------------------------------------------------------
# Bit level scanner for the 48 bit bzip2 magics
# ---------------------------------------------------------------------------

_BZIP_BLOCK_MAGIC = 0x314159265359
_BZIP_EOS_MAGIC = 0x177245385090
_BZIP_BLOCK_BYTES = _BZIP_BLOCK_MAGIC.to_bytes(6, "big")
_BZIP_EOS_BYTES = _BZIP_EOS_MAGIC.to_bytes(6, "big")


class _BitMarkerScanner:
    """Scan an MSB-first bit stream for 48 bit bzip2 marker words.

    Bzip2 block / EOS magics are packed at arbitrary *bit* offsets.  The
    scanner shifts each chunk by the eight possible sub-byte offsets using
    big integer arithmetic (C speed) and then locates the six byte marker in
    each shifted view with :meth:`bytes.find`.
    """

    def __init__(self) -> None:
        self._carry = b""
        self.total_bits = 0
        self._seen: set[int] = set()

    def feed(self, data: bytes) -> list[tuple[int, bool]]:
        """Return ``[(absolute_bit_position, is_block_magic), ...]``.

        EOS markers are reported as ``False``.
        """
        old_total = self.total_bits
        region = self._carry + data
        region_base = old_total - len(self._carry) * 8
        length = len(region)
        found: list[tuple[int, bool]] = []
        if length:
            region_value = int.from_bytes(region, "big")
            for shift in range(8):
                bit_count = 8 * length
                # Left shifting (then masking to the region width) realigns the
                # MSB-first stream to begin at bit ``shift``; padding lands at
                # the tail as the decoder expects.
                window = (region_value << shift) & ((1 << bit_count) - 1)
                view = window.to_bytes(length, "big")
                for marker, is_block in (
                    (_BZIP_BLOCK_BYTES, True),
                    (_BZIP_EOS_BYTES, False),
                ):
                    index = 0
                    while True:
                        at = view.find(marker, index)
                        if at < 0:
                            break
                        index = at + 1
                        bit_pos = at * 8 + shift
                        if bit_pos + 48 > 8 * length:
                            continue
                        absolute = region_base + bit_pos
                        # Accept new markers and markers straddling the
                        # boundary (they start in the carry bytes).
                        if absolute < old_total - 40:
                            continue
                        if absolute not in self._seen:
                            self._seen.add(absolute)
                            found.append((absolute, is_block))
        self._carry = region[-5:]
        self.total_bits = old_total + len(data) * 8
        found.sort()
        return found


def _extract_bits(raw_pre: Callable[[int, int], bytes], start_bit: int, bit_count: int) -> int:
    """Read ``bit_count`` MSB-first bits beginning at ``start_bit``."""
    if bit_count <= 0:
        return 0
    start_byte = start_bit >> 3
    skip = start_bit & 7
    end_byte = (start_bit + bit_count + 7) >> 3
    raw = raw_pre(start_byte, end_byte - start_byte)
    value = int.from_bytes(b"\xff" + raw, "big")
    total_bits = len(raw) * 8 - skip
    return (value >> (total_bits - bit_count)) & ((1 << bit_count) - 1)


def _build_synth_bzip(level: int, body: int, body_bits: int, crc: int) -> bytes:
    """Wrap a raw block bit string in a single block bzip2 stream.

    Bit padding must be *trailing* (the decoder starts reading the header at
    bit 0), so the integer is shifted left by the padding width.
    """
    header = int.from_bytes(b"BZh" + str(level).encode("ascii"), "big")
    eos = _BZIP_EOS_MAGIC
    value = (header << (body_bits + 80)) | (body << 80) | (eos << 32) | crc
    total_bits = body_bits + 32 + 48 + 32
    pad = (-total_bits) % 8
    return (value << pad).to_bytes((total_bits + 7) // 8, "big")


# ---------------------------------------------------------------------------
# Compression source
# ---------------------------------------------------------------------------

@dataclass
class _PendingBreaks:
    breaks: array.array = field(default_factory=lambda: array.array("Q"))


class CompressionSource(ABC):
    """Abstract random access source over a compressed file."""

    format = ""

    def __init__(
        self,
        identity: FileIdentity,
        cache: ContentAddressedCache | None = None,
        sidecars: SidecarStore | None = None,
    ) -> None:
        self.identity = identity
        self.compressed_size = identity.size
        self.cache = cache or ContentAddressedCache()
        self.sidecars = sidecars or SidecarStore()

        self.spool = SpoolFile()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._fatal: BaseException | None = None
        self._complete = False

        self._materialized = 0
        self._checkpoints: list[Checkpoint] = []
        self._breaks = _PendingBreaks()

        self._sidecar_breaks = self.sidecars.load_breaks(identity)
        self.sidecar_verified = self._sidecar_breaks is None
        self.sidecar_mismatch = False

        self._raw: _RawInput | None = None
        self._max_output = MAX_DECOMPRESSED_MB * 1024 * 1024

        # Neighbour prefetch for random reads ahead of the spool.
        self._prefetch_queue: "queue.Queue[tuple[int, int]]" = queue.Queue(maxsize=128)
        self._prefetch_thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the background materialiser (idempotent)."""
        with self._condition:
            if self._started or self._closed:
                return
            self._started = True
        self._thread = threading.Thread(
            target=self._run_materialiser, name=f"materialise:{self.format}:{self.identity.path}",
            daemon=True,
        )
        self._thread.start()
        self._prefetch_thread = threading.Thread(
            target=self._run_prefetch, name=f"prefetch:{self.format}", daemon=True
        )
        self._prefetch_thread.start()

    def start_and_wait(self, exit_event, min_bytes: int = 128 * 1024) -> bool:
        """Start and wait until the first segment is browsable.

        Returns ``False`` if ``exit_event`` was set before any data was ready.
        Raises if the stream is fatally corrupt / malicious.
        """
        self.start()
        with self._condition:
            while (
                self._materialized < min_bytes
                and not self._complete
                and self._fatal is None
                and not exit_event.is_set()
            ):
                self._condition.wait(0.2)
            if self._fatal is not None:
                raise self._fatal
            return self._materialized > 0 or self._complete

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._prefetch_queue.put((-1, -1))
        for thread in (self._thread, self._prefetch_thread):
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        if self._raw is not None:
            self._raw.close()
            self._raw = None
        self.spool.close()

    # -- properties ---------------------------------------------------------

    @property
    def materialized_size(self) -> int:
        return self._materialized

    @property
    def fileno(self) -> int:
        return self.spool.fileno

    @property
    def can_tail(self) -> bool:
        return False

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def progress_fraction(self) -> float:
        if not self.compressed_size:
            return 0.0 if not self._complete else 1.0
        return min(1.0, self._raw.bytes_read / self.compressed_size) if self._raw else 0.0

    @property
    def initial_breaks(self) -> list[int] | None:
        """Newline offsets loaded from a validated sidecar, if any."""
        return self._sidecar_breaks

    def get_breaks(self) -> list[int]:
        with self._condition:
            return list(self._breaks.breaks)

    def load_timestamp_sidecar(self) -> list[tuple[int, int, float]] | None:
        loaded = self.sidecars._read(self.identity, "timestamps")
        if loaded is None:
            return None
        header, _payload = loaded
        # Once the full digest is known it must match exactly; a fingerprint
        # match alone is not enough to serve cached timestamps.
        sha256 = self.identity.sha256
        if sha256 is not None and header.get("identity", {}).get("sha256") != sha256:
            return None
        return self.sidecars.load_timestamps(self.identity)

    def save_timestamp_sidecar(self, records: list[tuple[int, int, float]]) -> None:
        if not records:
            return
        identity = self.identity
        if identity.sha256 is None:
            return
        try:
            self.sidecars.save_timestamps(identity, records)
        except OSError:
            pass

    # -- waiting / scanning -------------------------------------------------

    def wait_done(self, cancel_event) -> bool:
        """Block until materialisation completes. ``False`` if cancelled."""
        with self._condition:
            while not self._complete and self._fatal is None:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                self._condition.wait(0.2)
            if self._fatal is not None:
                raise self._fatal
        return True

    def wait_for(self, logical_offset: int, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._materialized < logical_offset and self._fatal is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(0.2 if remaining is None else min(0.2, remaining))
            return self._fatal is None and self._materialized >= logical_offset

    def scan_line_breaks(
        self, cancel_event, batch_time: float = 0.25
    ) -> Iterable[tuple[int, list[int], float]]:
        """Forward, incremental newline scan over the growing spool.

        Yields ``(logical_position, breaks, progress_fraction)``.  The
        delivery cursor is local to each generator so tabs sharing one source
        can scan independently.
        """
        monotonic = time.monotonic
        batch: list[int] = []
        break_time = monotonic()
        position = 0
        delivered = 0
        while True:
            with self._condition:
                breaks = self._breaks.breaks
                if delivered < len(breaks):
                    batch.extend(breaks[delivered:])
                    delivered = len(breaks)
                position = self._materialized
                done = self._complete
                fatal = self._fatal
                fraction = self.progress_fraction
                flush = len(batch) >= 1000 and monotonic() - break_time > batch_time

            if batch and flush:
                yield position, batch, fraction
                batch = []
                break_time = monotonic()

            if cancel_event is not None and cancel_event.is_set():
                if batch:
                    yield position, batch, fraction
                return
            if fatal is not None or done:
                if batch:
                    yield position, batch, fraction
                return
            with self._condition:
                self._condition.wait(timeout=0.15)

    # -- reading ------------------------------------------------------------

    def read(self, start: int, end: int) -> bytes:
        """Read logical ``[start, end)`` (spool first, on demand otherwise)."""
        if end <= start:
            return b""
        if self._fatal is not None:
            return b""
        if end <= self._materialized:
            return self.spool.pread(start, end - start)
        # Wait a little for the background pass; fall back to direct decode.
        self.wait_for(end, timeout=0.5)
        if end <= self._materialized:
            return self.spool.pread(start, end - start)
        try:
            return self._read_on_demand(start, end)
        except CompressionError:
            if self._fatal is not None:
                return b""
            raise

    # -- materialiser plumbing ---------------------------------------------

    def _run_materialiser(self) -> None:
        try:
            self._raw = _RawInput(self.identity.path)
            self._materialise(self._raw)
            final_identity = self.identity.with_sha256(self._raw.hasher.hexdigest())
            with self._condition:
                # The compressed digest validates any previously loaded sidecar.
                self._verify_sidecar(final_identity)
                self.identity = final_identity
                self._complete = True
                self._condition.notify_all()
            self._persist_sidecar(final_identity)
        except BaseException as error:  # stored, surfaced via reads / waits
            with self._condition:
                self._fatal = error
                self._complete = True
                self._condition.notify_all()
        # Note: ``self._raw`` is deliberately kept open until :meth:`close` so
        # that on demand decoding can re-read compressed members / blocks.

    def _verify_sidecar(self, final_identity: FileIdentity) -> None:
        if self._sidecar_breaks is None:
            self.sidecar_verified = True
            return
        canonical = list(self._breaks.breaks)
        if canonical == self._sidecar_breaks:
            self.sidecar_verified = True
        else:
            self.sidecar_verified = False
            self.sidecar_mismatch = True
            self.sidecars.discard(final_identity)

    def _persist_sidecar(self, identity: FileIdentity) -> None:
        if self._fatal is not None:
            return
        try:
            with self._condition:
                breaks = array.array("Q", self._breaks.breaks)
            if self.sidecar_mismatch or self._sidecar_breaks is None or not self.sidecar_verified:
                self.sidecars.save_breaks(identity, breaks)
        except OSError:
            pass

    def _emit(self, data: bytes) -> None:
        """Append decoded bytes to the spool and record newline offsets."""
        if not data:
            return
        if self._materialized + len(data) > self._max_output:
            raise DecompressionBombError(
                f"decompressed size exceeds {self._max_output // (1024 * 1024)} MiB safety cap"
            )
        base = self._materialized
        offset = 0
        with self._condition:
            while True:
                at = data.find(b"\n", offset)
                if at < 0:
                    break
                self._breaks.breaks.append(base + at)
                offset = at + 1
            self.spool.append(data)
            self._materialized += len(data)
            self._condition.notify_all()

    def _add_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._condition:
            self._checkpoints.append(checkpoint)

    def _checkpoint_for(self, logical_offset: int) -> Checkpoint | None:
        with self._condition:
            checkpoints = list(self._checkpoints)
        for checkpoint in reversed(checkpoints):
            if checkpoint.logical_start <= logical_offset:
                return checkpoint
        return checkpoints[0] if checkpoints else None

    def _wait_for_checkpoint(self, logical_offset: int) -> Checkpoint:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            checkpoint = self._checkpoint_for(logical_offset)
            if checkpoint is not None:
                return checkpoint
            if self._fatal is not None or self._closed:
                raise CompressionError("checkpoint unavailable")
            time.sleep(0.1)
        raise CompressionError("timed out waiting for compression checkpoint")

    # -- on demand decoding -------------------------------------------------

    def _unit_crc(self, data: bytes) -> int:
        """Format native CRC of a decoded unit (gzip: zlib CRC32)."""
        return zlib.crc32(data) & 0xFFFFFFFF

    def _read_on_demand(self, start: int, end: int) -> bytes:
        # Random reads may arrive before (or without) background start.
        if not self._started and not self._closed:
            self.start()
        pieces: list[bytes] = []
        position = start
        while position < end:
            checkpoint = self._wait_for_checkpoint(position)
            key = self.cache.key(
                self.identity, checkpoint.index, self._unit_compressed(checkpoint)
            )
            unit = self.cache.get(
                key, self.identity, checkpoint.index, checkpoint.logical_start
            )
            if unit is None:
                unit = self._decode_unit(checkpoint)
                if self._unit_crc(unit) != checkpoint.crc:
                    # Defensive: decoders normally raise before this point.
                    raise CorruptCompressionError("CRC mismatch decoding on demand")
                if len(unit) != checkpoint.logical_length:
                    raise CorruptCompressionError("length mismatch decoding on demand")
                self.cache.put(
                    key, self.identity, checkpoint.index, checkpoint.logical_start, unit
                )
            elif self._unit_crc(unit) != checkpoint.crc:
                # Poisoned cache entry that somehow survived content validation.
                raise CorruptCompressionError("CRC mismatch on cached decompressed unit")
            if position == start:
                self._schedule_prefetch(checkpoint.index)
            unit_end = checkpoint.logical_start + len(unit)
            slice_start = position - checkpoint.logical_start
            slice_end = min(end, unit_end) - checkpoint.logical_start
            pieces.append(unit[slice_start:slice_end])
            position += slice_end - slice_start
            if unit_end <= position:
                # Stop when no later unit has been discovered yet; at EOF the
                # short result is the correct read, near the frontier the spool
                # will serve the rest shortly.
                next_checkpoint = self._checkpoint_for(position)
                if next_checkpoint is None or next_checkpoint.index <= checkpoint.index:
                    break
        return b"".join(pieces)

    def _schedule_prefetch(self, index: int) -> None:
        for neighbour in (index - 1, index + 1):
            if neighbour < 0:
                continue
            try:
                self._prefetch_queue.put_nowait((neighbour, index))
            except Exception:
                pass

    def _run_prefetch(self) -> None:
        queued: set[int] = set()
        while not self._closed:
            try:
                index, _near = self._prefetch_queue.get(timeout=0.25)
            except Exception:
                continue
            if index == -1 or self._closed:
                return
            if index in queued:
                continue
            queued.add(index)
            try:
                with self._condition:
                    checkpoints = list(self._checkpoints)
                if index >= len(checkpoints):
                    continue
                checkpoint = checkpoints[index]
                compressed = self._unit_compressed(checkpoint)
                key = self.cache.key(self.identity, index, compressed)
                if self.cache.get(
                    key, self.identity, index, checkpoint.logical_start
                ) is None:
                    unit = self._decode_unit(checkpoint)
                    self.cache.put(
                        key, self.identity, index, checkpoint.logical_start, unit
                    )
            except Exception:
                # Prefetch is best effort.
                pass

    # -- subclass hooks -----------------------------------------------------

    @abstractmethod
    def _materialise(self, raw: _RawInput) -> None:
        """Decode the whole stream, emitting segments and checkpoints."""

    @abstractmethod
    def _unit_compressed(self, checkpoint: Checkpoint) -> bytes:
        """Return the raw compressed bytes covering a checkpoint unit."""

    @abstractmethod
    def _decode_unit(self, checkpoint: Checkpoint) -> bytes:
        """Decode a single member / block (dictionary reset at the boundary)."""


# ---------------------------------------------------------------------------
# Gzip
# ---------------------------------------------------------------------------

class GzipSource(CompressionSource):
    format = "gzip"

    def _materialise(self, raw: _RawInput) -> None:
        decompressor = zlib.decompressobj(31)
        member_start = 0
        member_logical_start = 0
        member_length = 0
        member_crc = 0
        member_index = 0
        # ``closed`` is True once the current member has been fully terminated
        # and the next file bytes must start a fresh member.
        closed = True

        while True:
            chunk = raw.read(READ_SIZE)
            if not chunk:
                break
            closed = False
            try:
                inflated = decompressor.decompress(chunk)
            except zlib.error as error:
                raise CorruptCompressionError(f"invalid gzip stream: {error}") from error
            if inflated:
                member_length += len(inflated)
                member_crc = zlib.crc32(inflated, member_crc)
                # Enforce the ratio while the member is still streaming, not
                # just at its trailer, so a malicious member can never emit
                # more than ratio * compressed-read bytes before rejection.
                if member_length and (
                    (raw.bytes_read - member_start) * GZIP_MAX_RATIO < member_length
                ):
                    raise DecompressionBombError(
                        "gzip member compression ratio exceeds safety cap"
                    )
                self._emit(inflated)

            while decompressor.eof:
                unused = decompressor.unused_data
                member_end = raw.bytes_read - len(unused)
                self._validate_gzip_member(
                    raw, member_start, member_end, member_length, member_crc
                )
                self._add_checkpoint(
                    Checkpoint(
                        member_index,
                        member_start,
                        member_end,
                        member_logical_start,
                        member_length,
                        member_crc & 0xFFFFFFFF,
                    )
                )
                member_index += 1
                member_logical_start += member_length
                # The next member (if any) needs a decompressor with a reset
                # DEFLATE dictionary.
                decompressor = zlib.decompressobj(31)
                if not unused:
                    closed = True
                    break
                if len(unused) < 3 or unused[:3] != b"\x1f\x8b\x08":
                    raise CorruptCompressionError("trailing garbage after gzip member")
                try:
                    inflated = decompressor.decompress(unused)
                except zlib.error as error:
                    raise CorruptCompressionError(
                        f"invalid gzip member: {error}"
                    ) from error
                member_start = member_end
                member_length = len(inflated)
                member_crc = zlib.crc32(inflated, 0)
                self._emit(inflated)
                if not decompressor.eof:
                    break

        if not closed:
            raise CorruptCompressionError("gzip stream is truncated")
        if member_index == 0:
            raise CorruptCompressionError("not a gzip file (no member found)")

    @staticmethod
    def _validate_gzip_member(
        raw: _RawInput,
        start: int,
        end: int,
        logical_length: int,
        crc: int,
    ) -> None:
        if end - start < 18:
            raise CorruptCompressionError("gzip member is impossibly short")
        if logical_length and (end - start) * GZIP_MAX_RATIO < logical_length:
            raise DecompressionBombError("gzip member compression ratio exceeds safety cap")
        trailer = raw.pread(end - 8, 8)
        if len(trailer) != 8:
            raise CorruptCompressionError("gzip trailer is missing")
        stored_crc, stored_size = struct.unpack("<II", trailer)
        if stored_crc != (crc & 0xFFFFFFFF):
            raise CorruptCompressionError("gzip CRC32 mismatch (corrupt member)")
        if stored_size != logical_length & 0xFFFFFFFF:
            raise CorruptCompressionError("gzip ISIZE mismatch (corrupt member)")

    def _unit_compressed(self, checkpoint: Checkpoint) -> bytes:
        raw = self._raw
        if raw is None:
            # Source closed: reopen the immutable regular file for one decode.
            raw = _RawInput(self.identity.path)
            try:
                return raw.pread(checkpoint.start, checkpoint.end - checkpoint.start)
            finally:
                raw.close()
        return raw.pread(checkpoint.start, checkpoint.end - checkpoint.start)

    def _decode_unit(self, checkpoint: Checkpoint) -> bytes:
        compressed = self._unit_compressed(checkpoint)
        decompressor = zlib.decompressobj(31)
        try:
            unit = decompressor.decompress(compressed)
            unit += decompressor.flush()
        except zlib.error as error:
            raise CorruptCompressionError(f"invalid gzip member: {error}") from error
        if not decompressor.eof or decompressor.unused_data:
            raise CorruptCompressionError("gzip member boundary not recovered")
        if len(unit) != checkpoint.logical_length:
            raise CorruptCompressionError("gzip member length mismatch")
        return unit


# ---------------------------------------------------------------------------
# Bzip2
# ---------------------------------------------------------------------------

class Bzip2Source(CompressionSource):
    """Random access source for bzip2 streams.

    Bzip2 streams consist of independent, CRC protected *blocks* whose 48 bit
    start magics live at arbitrary *bit* offsets.  Blocks can therefore be
    recovered by extracting the bit range between two magics and wrapping it
    in a synthetic one-block ``BZh<level>`` stream.  False marker hits inside
    block payloads (~ 2^-48) are rejected because their synthetic stream fails
    to decode / CRC check.
    """

    format = "bzip2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Validated stream header bit positions -> compression level.
        self._stream_headers: dict[int, int] = {}

    # -- materialisation ----------------------------------------------------

    def _materialise(self, raw: _RawInput) -> None:
        header = raw.pread(0, 4)
        if len(header) < 4 or header[:3] != b"BZh" or header[3] not in b"123456789":
            raise CorruptCompressionError("not a bzip2 file (bad BZh header)")
        self._stream_headers = {0: header[3] - 0x30}

        scanner = _BitMarkerScanner()
        pending: list[int] = []  # candidate block starts awaiting a boundary
        stream_crcs: list[int] = []
        deferred_eos: list[int] = []
        block_index = 0
        logical = 0
        got_data = False

        while True:
            chunk = raw.read(READ_SIZE)
            if not chunk:
                break
            got_data = True
            horizon_bits = raw.bytes_read * 8
            # Finalise EOS markers from earlier feeds before any new block
            # markers, otherwise the next stream's block CRCs would pollute
            # the previous stream's CRC list.
            for eos_bit in list(deferred_eos):
                if self._try_finalise_eos(
                    raw, eos_bit, stream_crcs, horizon_bits, at_eof=False
                ):
                    deferred_eos.remove(eos_bit)
            for bit_position, is_block in scanner.feed(chunk):
                winner = self._close_candidates(raw, pending, bit_position)
                if is_block:
                    if winner is not None:
                        decoded, start = winner
                        pending.remove(start)
                        block_index, logical = self._emit_block(
                            raw, start, bit_position, decoded, block_index, logical,
                            stream_crcs,
                        )
                    pending.append(bit_position)
                elif winner is None:
                    # A false EOS marker inside a payload: nothing decodes
                    # ending here, ignore it and wait for the true EOS.
                    continue
                else:
                    decoded, start = winner
                    pending = []
                    block_index, logical = self._emit_block(
                        raw, start, bit_position, decoded, block_index, logical,
                        stream_crcs,
                    )
                    # Finalise immediately so the CRC list is cleared before
                    # the next concatenated stream's blocks are emitted.
                    if not self._try_finalise_eos(
                        raw, bit_position, stream_crcs, horizon_bits, at_eof=False
                    ):
                        deferred_eos.append(bit_position)

        if not got_data:
            raise CorruptCompressionError("empty bzip2 file")
        horizon_bits = raw.bytes_read * 8
        for eos_bit in list(deferred_eos):
            if not self._try_finalise_eos(
                raw, eos_bit, stream_crcs, horizon_bits, at_eof=True
            ):
                raise CorruptCompressionError("truncated bzip2 EOS trailer")
        if pending:
            raise CorruptCompressionError("bzip2 stream is truncated (no EOS magic)")

        # Independent streaming verification of the complete concatenation.
        if os.environ.get("TOOLONG_BZIP2_VERIFY", "1") != "0":
            self._verify_streaming(raw, logical)

    def _emit_block(
        self,
        raw: _RawInput,
        start_bit: int,
        end_bit: int,
        decoded: bytes,
        block_index: int,
        logical: int,
        stream_crcs: list[int],
    ) -> tuple[int, int]:
        level = self._level_at(raw, start_bit)
        crc = self._block_crc(raw, start_bit)
        self._add_checkpoint(
            Checkpoint(
                block_index,
                start_bit,
                end_bit,
                logical,
                len(decoded),
                crc,
                level=level,
            )
        )
        self._emit(decoded)
        stream_crcs.append(crc)
        return block_index + 1, logical + len(decoded)

    def _close_candidates(
        self, raw: _RawInput, candidates: list[int], end_bit: int
    ) -> tuple[bytes, int] | None:
        """Try candidate block starts against a known boundary.

        The first candidate that decodes is the real block; 48 bit marker hits
        inside payload data simply fail to decode and are dropped later.
        """
        for start_bit in candidates:
            decoded = self._try_decode(raw, start_bit, end_bit)
            if decoded is not None:
                return decoded, start_bit
        return None

    def _try_decode(
        self, raw: _RawInput, start_bit: int, end_bit: int
    ) -> bytes | None:
        try:
            level = self._level_at(raw, start_bit)
        except CorruptCompressionError:
            return None
        crc = self._block_crc(raw, start_bit)
        body = _extract_bits(raw.pread, start_bit, end_bit - start_bit)
        synth = _build_synth_bzip(level, body, end_bit - start_bit, crc)
        try:
            decoder = bz2.BZ2Decompressor()
            decoded = decoder.decompress(synth)
        except OSError:
            return None
        if not decoder.eof or decoder.unused_data:
            return None
        if bzip2_crc32(decoded) != crc:
            # False 48 bit marker hit (or corrupted embedded CRC): not a real
            # block, so the caller drops the candidate instead of bombing.
            return None
        compressed_bits = end_bit - start_bit
        if len(decoded) and compressed_bits * BZIP2_MAX_RATIO < len(decoded) * 8:
            raise DecompressionBombError("bzip2 block ratio exceeds safety cap")
        return decoded

    @staticmethod
    def _unit_crc(data: bytes) -> int:
        return bzip2_crc32(data)

    @staticmethod
    def _block_crc(raw: _RawInput, start_bit: int) -> int:
        return _extract_bits(raw.pread, start_bit + 48, 32)

    def _level_at(self, raw: _RawInput, bit_position: int) -> int:
        candidate: int | None = None
        for header_bit in sorted(self._stream_headers):
            if header_bit <= bit_position:
                candidate = header_bit
            else:
                break
        if candidate is None:
            raise CorruptCompressionError("bzip2 block outside any stream")
        return self._stream_headers[candidate]

    def _try_finalise_eos(
        self,
        raw: _RawInput,
        eos_bit: int,
        stream_crcs: list[int],
        horizon_bits: int,
        at_eof: bool,
    ) -> bool:
        """Validate an EOS magic. Returns True when finalised."""
        end_bit = eos_bit + 48 + 32
        next_byte_bit = (end_bit + 7) & ~7
        next_byte = next_byte_bit >> 3
        if horizon_bits < end_bit:
            if at_eof:
                raise CorruptCompressionError("truncated bzip2 EOS trailer")
            return False
        expected = _extract_bits(raw.pread, eos_bit + 48, 32)
        combined = 0
        for block_crc in stream_crcs:
            combined = (((combined << 1) | (combined >> 31)) & 0xFFFFFFFF) ^ block_crc
        if expected != combined:
            raise CorruptCompressionError("bzip2 stream CRC mismatch")
        if next_byte_bit > end_bit:
            padding = _extract_bits(raw.pread, end_bit, next_byte_bit - end_bit)
            if padding != 0:
                raise CorruptCompressionError("non-zero bzip2 tail padding")
        available = (horizon_bits >> 3) - next_byte
        if available == 0 and not at_eof:
            # May be EOF or the start of a concatenated stream.
            return False
        trailing = raw.pread(next_byte, 4)
        if trailing:
            if len(trailing) < 4 or trailing[:3] != b"BZh" or trailing[3] not in b"123456789":
                raise CorruptCompressionError("garbage after bzip2 stream")
            self._stream_headers[next_byte_bit] = trailing[3] - 0x30
        stream_crcs.clear()
        return True

    def _verify_streaming(self, raw: _RawInput, expected_length: int) -> None:
        decoder = bz2.BZ2Decompressor()
        offset = 0
        produced = 0
        while offset < self.compressed_size:
            chunk = raw.pread(offset, READ_SIZE)
            if not chunk:
                break
            offset += len(chunk)
            try:
                produced += len(decoder.decompress(chunk))
                if decoder.eof and decoder.unused_data:
                    leftover = decoder.unused_data
                    decoder = bz2.BZ2Decompressor()
                    produced += len(decoder.decompress(leftover))
            except OSError as error:
                raise CorruptCompressionError(
                    f"bzip2 verification failed: {error}"
                ) from error
        if produced != expected_length:
            raise CorruptCompressionError("bzip2 output length mismatch during verification")

    # -- on demand ----------------------------------------------------------

    def _unit_compressed(self, checkpoint: Checkpoint) -> bytes:
        start_byte = checkpoint.start >> 3
        end_byte = (checkpoint.end + 7) >> 3
        raw = self._raw
        opened = False
        if raw is None:
            raw = _RawInput(self.identity.path)
            opened = True
        try:
            return raw.pread(start_byte, end_byte - start_byte + 1)
        finally:
            if opened:
                raw.close()

    def _decode_unit(self, checkpoint: Checkpoint) -> bytes:
        raw = self._raw
        opened = False
        if raw is None:
            raw = _RawInput(self.identity.path)
            opened = True
        try:
            decoded = self._try_decode(raw, checkpoint.start, checkpoint.end)
        finally:
            if opened:
                raw.close()
        if decoded is None:
            raise CorruptCompressionError("bzip2 block boundary not recovered")
        if len(decoded) != checkpoint.logical_length:
            raise CorruptCompressionError("bzip2 block length mismatch")
        return decoded


# ---------------------------------------------------------------------------
# Process local registry: one shared source per compressed file per process
# ---------------------------------------------------------------------------

class CompressionSourceRegistry:
    """Shares a single source between tabs and owns its lifecycle."""

    _CLASSES = {"gzip": GzipSource, "bzip2": Bzip2Source}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: dict[str, tuple[CompressionSource, int]] = {}

    def acquire(self, path: str, encoding: str) -> CompressionSource:
        cls = self._CLASSES.get(encoding)
        if cls is None:
            raise CompressionError(f"unsupported compression {encoding!r}")
        resolved = str(Path(path).resolve())
        identity = compute_identity(resolved, encoding)
        with self._lock:
            entry = self._sources.get(identity.stable_id)
            if entry is not None:
                source, refs = entry
                self._sources[identity.stable_id] = (source, refs + 1)
                return source
            source = cls(identity)
            self._sources[identity.stable_id] = (source, 1)
            return source

    def release(self, source: CompressionSource) -> None:
        with self._lock:
            entry = self._sources.get(source.identity.stable_id)
            if entry is None:
                return
            current, refs = entry
            if current is not source:
                return
            refs -= 1
            if refs > 0:
                self._sources[source.identity.stable_id] = (source, refs)
                return
            self._sources.pop(source.identity.stable_id, None)
        source.close()

    def close_all(self) -> None:
        with self._lock:
            sources = [source for source, _ in self._sources.values()]
            self._sources.clear()
        for source in sources:
            source.close()


_registry: CompressionSourceRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> CompressionSourceRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = CompressionSourceRegistry()
        return _registry
