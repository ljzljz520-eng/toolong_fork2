from __future__ import annotations

from datetime import datetime
import os
import mmap
import mimetypes
import platform
import time
from pathlib import Path
from typing import IO, Iterable
from threading import Event, Lock

import rich.repr

from toolong.format_parser import FormatParser, ParseResult
from toolong.timestamps import TimestampScanner


IS_WINDOWS = platform.system() == "Windows"


class LogError(Exception):
    """An error related to logs."""


@rich.repr.auto(angular=True)
class LogFile:
    """A single log file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.name = self.path.name
        self.file: IO[bytes] | None = None
        self._size = 0
        self.can_tail = False
        self.timestamp_scanner = TimestampScanner()
        self.format_parser = FormatParser()
        self._lock = Lock()
        self._source = None  # CompressionSource for compressed files

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.name
        yield "size", self.size

    @property
    def is_open(self) -> bool:
        return self.file is not None or self._source is not None

    @property
    def fileno(self) -> int:
        if self._source is not None:
            return self._source.fileno
        assert self.file is not None
        return self.file.fileno()

    @property
    def size(self) -> int:
        if self._source is not None:
            return self._source.materialized_size
        return self._size

    @property
    def is_compressed(self) -> bool:
        _, encoding = mimetypes.guess_type(self.path.name, strict=False)
        return encoding in ("gzip", "bzip2")

    @property
    def source(self):
        """The backing CompressionSource, if this is a compressed file."""
        return self._source

    def parse(self, line: str) -> ParseResult:
        """Parse a line."""
        return self.format_parser.parse(line)

    def get_create_time(self) -> datetime | None:
        try:
            stat_result = self.path.stat()
        except Exception:
            return None
        try:
            # This works on Mac
            create_time_seconds = stat_result.st_birthtime
        except AttributeError:
            # No birthtime for Linux, so we assume the epoch start
            return datetime.fromtimestamp(0)
        timestamp = datetime.fromtimestamp(create_time_seconds)
        return timestamp

    def open(self, exit_event: Event) -> bool:
        # Check for compressed files
        _, encoding = mimetypes.guess_type(self.path.name, strict=False)

        # Open compressed files
        if encoding in ("gzip", "bzip2"):
            # An empty file carries no gzip / bzip2 header; treat it like an
            # ordinary (empty) file rather than reporting a corrupt stream.
            try:
                if self.path.stat().st_size == 0:
                    encoding = ""
            except OSError:
                pass
            if encoding:
                return self.open_compressed(exit_event, encoding)

        # Open uncompressed file
        self.file = open(self.path, "rb", buffering=0)

        self.file.seek(0, os.SEEK_END)
        self._size = self.file.tell()
        self.can_tail = True
        return True

    def open_compressed(self, exit_event: Event, encoding: str) -> bool:
        from toolong.compression import get_registry

        registry = get_registry()
        source = registry.acquire(str(self.path.resolve()), encoding)
        try:
            ready = source.start_and_wait(exit_event)
        except BaseException:
            registry.release(source)
            raise
        if not ready:
            registry.release(source)
            return False
        self._source = source
        self.can_tail = False
        return True

    def close(self) -> None:
        if self._source is not None:
            from toolong.compression import get_registry

            source = self._source
            self._source = None
            get_registry().release(source)
        if self.file is not None:
            self.file.close()
            self.file = None

    if IS_WINDOWS:

        def get_raw(self, start: int, end: int) -> bytes:
            if self._source is not None:
                return self._source.read(start, end)
            with self._lock:
                if start >= end or self.file is None:
                    return b""
                position = os.lseek(self.fileno, 0, os.SEEK_CUR)
                try:
                    os.lseek(self.fileno, start, os.SEEK_SET)
                    return os.read(self.fileno, end - start)
                finally:
                    os.lseek(self.fileno, position, os.SEEK_SET)

    else:

        def get_raw(self, start: int, end: int) -> bytes:
            if self._source is not None:
                return self._source.read(start, end)
            if start >= end or self.file is None:
                return b""
            return os.pread(self.fileno, end - start, start)

    def get_line(self, start: int, end: int) -> str:
        return (
            self.get_raw(start, end)
            .decode("utf-8", errors="replace")
            .strip("\n\r")
            .expandtabs(4)
        )

    def scan_line_breaks(
        self, batch_time: float = 0.25
    ) -> Iterable[tuple[int, list[int]]]:
        """Scan the file for line breaks.

        Args:
            batch_time: Time to group the batches.

        Returns:
            An iterable of tuples, containing the scan position and a list of offsets of new lines.
        """
        fileno = self.fileno
        size = self.size
        if not size:
            return
        if IS_WINDOWS:
            log_mmap = mmap.mmap(fileno, size, access=mmap.ACCESS_READ)
        else:
            log_mmap = mmap.mmap(fileno, size, prot=mmap.PROT_READ)
        try:
            rfind = log_mmap.rfind
            position = size
            batch: list[int] = []
            append = batch.append
            get_length = batch.__len__
            monotonic = time.monotonic
            break_time = monotonic()

            if log_mmap[-1] != "\n":
                batch.append(position)

            while (position := rfind(b"\n", 0, position)) != -1:
                append(position)
                if get_length() % 1000 == 0 and monotonic() - break_time > batch_time:
                    break_time = monotonic()
                    yield (position, batch)
                    batch = []
                    append = batch.append
            yield (0, batch)
        finally:
            log_mmap.close()

    def scan_compressed_line_breaks(
        self, cancel_event
    ) -> Iterable[tuple[int, list[int], float]]:
        """Forward newline scan backed by the CompressionSource.

        Yields ``(materialised_position, newline_offsets, progress_fraction)``
        tuples as decompression progresses.
        """
        source = self._source
        if source is None:
            return
        yield from source.scan_line_breaks(cancel_event)

    def scan_timestamps(
        self, batch_time: float = 0.25, cancel_event=None
    ) -> Iterable[list[tuple[int, int, float]]]:
        if self._source is not None:
            yield from self._scan_timestamps_compressed(
                batch_time=batch_time, cancel_event=cancel_event
            )
            return
        size = self.size
        if not size:
            return
        fileno = self.fileno
        if IS_WINDOWS:
            log_mmap = mmap.mmap(fileno, size, access=mmap.ACCESS_READ)
        else:
            log_mmap = mmap.mmap(fileno, size, prot=mmap.PROT_READ)

        monotonic = time.monotonic
        scan_time = monotonic()
        scan = self.timestamp_scanner.scan
        line_no = 0
        position = 0
        results: list[tuple[int, int, float]] = []
        append = results.append
        get_length = results.__len__
        while line_bytes := log_mmap.readline():
            line = line_bytes.decode("utf-8", errors="replace")
            timestamp = scan(line)
            position += len(line_bytes)
            append((line_no, position, timestamp.timestamp() if timestamp else 0.0))
            line_no += 1
            if (
                results
                and get_length() % 1000 == 0
                and monotonic() - scan_time > batch_time
            ):
                scan_time = monotonic()
                yield results
                results = []
                append = results.append
        if results:
            yield results

    # -- compressed timestamp scan -----------------------------------------

    def _scan_timestamps_compressed(
        self, batch_time: float = 0.25, cancel_event=None
    ) -> Iterable[list[tuple[int, int, float]]]:
        from toolong.compression import MAX_SIDECAR_LINES

        source = self._source
        assert source is not None
        # Timestamps describe the whole file, so the background materialiser
        # must finish (with full CRC validation) first.
        try:
            if not source.wait_done(cancel_event):
                return
        except Exception:
            # Fatal corruption: the caller (merge) gets no records for file.
            return
        if source.fatal_error is not None or source.materialized_size == 0:
            return
        size = source.materialized_size

        cached = source.load_timestamp_sidecar()
        if cached is not None:
            batch: list[tuple[int, int, float]] = []
            for record in cached:
                batch.append(record)
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        monotonic = time.monotonic
        scan_time = monotonic()
        scan = self.timestamp_scanner.scan
        line_no = 0
        position = 0
        results: list[tuple[int, int, float]] = []
        to_persist: list[tuple[int, int, float]] = []

        def process(raw_line: bytes) -> None:
            nonlocal line_no, position
            line = raw_line.decode("utf-8", errors="replace")
            timestamp = scan(line)
            position += len(raw_line)
            record = (line_no, position, timestamp.timestamp() if timestamp else 0.0)
            results.append(record)
            if line_no < MAX_SIDECAR_LINES:
                to_persist.append(record)
            line_no += 1

        read_size = 1024 * 1024
        offset = 0
        carry = b""
        while offset < size:
            chunk = source.spool.pread(offset, min(read_size, size - offset))
            if not chunk:
                break
            offset += len(chunk)
            data = carry + chunk
            search_from = 0
            while True:
                newline = data.find(b"\n", search_from)
                if newline < 0:
                    carry = data[search_from:]
                    break
                process(data[search_from : newline + 1])
                search_from = newline + 1
                if (
                    results
                    and len(results) % 1000 == 0
                    and monotonic() - scan_time > batch_time
                ):
                    scan_time = monotonic()
                    yield results
                    results = []
            if cancel_event is not None and cancel_event.is_set():
                return
        if carry:
            process(carry)
        if results:
            yield results
        if to_persist and line_no < MAX_SIDECAR_LINES:
            source.save_timestamp_sidecar(to_persist)
