import contextlib
import email.utils
import enum
import glob
import hashlib
import http.client
import json
import logging
import math
import os
import re
import secrets
import shutil
import ssl
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import certifi

from pyrosm import __version__
from pyrosm._log import log_event
from pyrosm.exceptions import DownloadError, RequestRefusedError
from pyrosm.utils import _file_identity
from pyrosm.utils.file_lock import exclusive_lock
from pyrosm.utils.progress import reporting, validate_progress

USER_AGENT = "pyrosm/%s (+https://github.com/pyrosm/pyrosm)" % __version__

logger = logging.getLogger(__name__)

_FETCH_ERRORS = (OSError, http.client.HTTPException)
_RETRY_STATUSES = (408, 425, 429)
_TIMEOUT = 60
_ATTEMPTS = 3
_BACKOFF = 1.0
_MAX_RETRY_AFTER = 60
_CHUNK = 1 << 18
# Headers that belong to the URL they were given for and are not sent on after a redirect.
_ORIGIN_BOUND = ("authorization", "cookie", "proxy-authorization", "host")
_STRONG_ETAG = re.compile(r'"[\x21\x23-\x7e\x80-\xff]*"')
_CONTENT_RANGE = re.compile(r"bytes (\d{1,19})-(\d{1,19})/(\d{1,19})")
# The full downloads of one URL that 24 hours allow; the next one is refused.
_FULL_DOWNLOADS_A_DAY = 3
_DAY = 24 * 3600
_RESPONSIBLY = "https://pyrosm.readthedocs.io/en/stable/download_responsibly.html"
_sleep = time.sleep


def _timed(connection_class, read_timeout):
    """``connection_class`` with ``read_timeout`` set on its socket once it is connected."""

    class Connection(connection_class):
        def connect(self):
            super().connect()
            self.sock.settimeout(read_timeout)

    return Connection


class _HTTPSHandler(urllib.request.HTTPSHandler):
    """Opens HTTPS connections that use one timeout to connect and another to read."""

    def __init__(self, context, read_timeout):
        super().__init__(context=context)
        self.tls_context = context
        self.connection = _timed(http.client.HTTPSConnection, read_timeout)

    def https_open(self, req):
        return self.do_open(self.connection, req, context=self.tls_context)


class _HTTPHandler(urllib.request.HTTPHandler):
    """Opens HTTP connections that use one timeout to connect and another to read."""

    def __init__(self, read_timeout):
        super().__init__()
        self.connection = _timed(http.client.HTTPConnection, read_timeout)

    def http_open(self, req):
        return self.do_open(self.connection, req)


def open_url(url, method="GET", headers=None, timeout=None, opener=None):
    """Open ``url`` with pyrosm's User-Agent and certifi's CA bundle.

    ``headers`` are added to (or override) the default ``User-Agent`` header; credentials,
    cookies and ``Host`` are not sent on to the target of a redirect. ``timeout`` (in
    seconds) limits the connect and each read; a ``(connect, read)`` pair sets them apart;
    ``None`` waits indefinitely. ``opener`` is an object with ``open(request, timeout=...)``,
    such as ``urllib.request.build_opener(...)``, that makes the request instead of pyrosm; it
    takes one timeout, so a pair with an opener raises ``ValueError``. Returns the response,
    usable as a context manager.
    """
    split = isinstance(timeout, tuple)
    if split and opener is not None:
        raise ValueError(
            "A (connect, read) timeout needs pyrosm's own opener; give one number with "
            "an opener."
        )
    connect, read = timeout if split else (timeout, timeout)
    headers = {"User-Agent": USER_AGENT, **(headers or {})}
    bound = {k: v for k, v in headers.items() if k.lower() in _ORIGIN_BOUND}
    request = urllib.request.Request(
        url,
        headers={k: v for k, v in headers.items() if k not in bound},
        method=method,
    )
    for name, value in bound.items():
        request.add_unredirected_header(name, value)
    if opener is not None:
        return opener.open(request, timeout=connect)
    # Build the HTTPS context from certifi's CA bundle instead of the OS trust store. On
    # Windows, loading the system certificate store can raise ssl.SSLError [ASN1:
    # NOT_ENOUGH_DATA] (a CPython bug triggered by a malformed entry in the store); certifi
    # avoids it and works the same across platforms.
    context = ssl.create_default_context(cafile=certifi.where())
    if split:
        opener = urllib.request.build_opener(
            _HTTPSHandler(context, read), _HTTPHandler(read)
        )
        return opener.open(request, timeout=connect)
    if timeout is None:
        return urllib.request.urlopen(request, context=context)
    return urllib.request.urlopen(request, context=context, timeout=timeout)


class _Net:
    """The caller's network options for pyrosm's requests: extra headers, the timeout and an
    opener (see :func:`open_url`)."""

    def __init__(self, headers=None, timeout=_TIMEOUT, opener=None):
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.opener = opener

    def open(self, url, method="GET", headers=None):
        """Open ``url``; pyrosm's own ``headers`` win over caller headers of the same name,
        in any letter case."""
        own = headers or {}
        names = {name.lower() for name in own}
        merged = {k: v for k, v in self.headers.items() if k.lower() not in names}
        return open_url(url, method, {**merged, **own}, self.timeout, self.opener)


def _content_length(value):
    """A ``Content-Length`` header value as an int in ``[0, 2**63)``, or ``None`` when it is
    missing or not a plain decimal number in that range."""
    if not value or not (value.isascii() and value.isdigit()) or len(value) > 19:
        return None
    size = int(value)
    return size if size < 2**63 else None


def _retry_after(error):
    """Seconds the ``Retry-After`` header of an ``HTTPError`` asks to wait, or ``None`` when it
    has none or it cannot be read. An HTTP-date is converted to seconds from now; a number too
    long to be a sensible wait counts as infinite."""
    value = (getattr(error, "headers", None) or {}).get("Retry-After")
    if not value:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        value = value.lstrip("0") or "0"
        return int(value) if len(value) <= 9 else float("inf")
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _retryable(error):
    """Whether ``error`` is a network error or an HTTP status worth another attempt."""
    if isinstance(error, HTTPError):
        return error.code in _RETRY_STATUSES or 500 <= error.code <= 599
    return isinstance(error, _FETCH_ERRORS)


def _retry(fetch, attempts=_ATTEMPTS):
    """Return ``fetch()``, calling it up to ``attempts`` times while it fails with a network
    error or HTTP status 408, 425, 429 or 5xx.

    Between attempts it waits the ``Retry-After`` the server sent, or else 1 s, 2 s, ... A
    ``Retry-After`` above 60 s is not waited for: the error is raised at once. Other HTTP errors
    and the last failure propagate.
    """
    attempt = 0
    while True:
        try:
            return fetch()
        except _FETCH_ERRORS as e:
            wait = None
            if isinstance(e, HTTPError):
                e.close()  # its response body is a temporary file
                wait = _retry_after(e)
            if not _retryable(e) or attempt + 1 >= attempts:
                raise
            if wait is None:
                wait = _BACKOFF * 2**attempt
            elif wait > _MAX_RETRY_AFTER:
                raise
            _sleep(wait)
            attempt += 1


def _validator(headers):
    """The ``If-Range`` value that pins a resumed download to the copy being downloaded.

    A strong ``ETag`` that follows the entity-tag grammar, else a ``Last-Modified`` at least one
    second older than the response's ``Date`` (strong by RFC 9110, section 8.8.2.2), written
    back as a GMT HTTP-date; ``None`` when there is neither.
    """
    etag = (headers.get("ETag") or "").strip()
    if _STRONG_ETAG.fullmatch(etag):
        return etag
    try:
        modified = email.utils.parsedate_to_datetime(headers.get("Last-Modified"))
        date = email.utils.parsedate_to_datetime(headers.get("Date"))
        if modified.tzinfo is not None and (date - modified).total_seconds() >= 1:
            return email.utils.format_datetime(modified.astimezone(timezone.utc), True)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _encoded(headers):
    """Whether a response body has a content coding, so its bytes are not the file's."""
    coding = (headers.get("Content-Encoding") or "identity").strip().lower()
    return coding != "identity"


class _NotModified(Exception):
    """The server answered a conditional request with 304: the local copy is current."""


class _Refused(Exception):
    """A full download that the download log refuses (see :func:`_download_log`) was about
    to be asked for or to begin."""


class _Transfer:
    """One file downloaded into ``out_file``, over as many attempts as it takes.

    An attempt after a dropped connection asks for the rest of the file with ``Range`` and
    ``If-Range`` when the server gave a strong validator, and starts over otherwise. After a
    partial answer that does not fit the copy, the rest of the call starts over every time.
    ``progress``, when given, is called as ``progress(written, total)`` when the server
    starts sending the file and after each chunk. With ``stored`` (the record of the copy
    downloaded before, see :func:`_stored_validators`) its ``If-None-Match`` /
    ``If-Modified-Since`` go with each request until the server sends the file; a 304 from
    the URL that served that copy raises :class:`_NotModified`, one from another URL makes
    the request again without them. ``etag``, ``last_modified`` and ``served`` (the URL
    that sent it) describe the full answer. With ``refuse_full``, :class:`_Refused` is
    raised instead of asking for the file in full or taking a full answer.
    """

    def __init__(
        self,
        url,
        filename,
        out_file,
        net,
        progress=None,
        stored=None,
        refuse_full=False,
    ):
        self.url = url
        self.filename = filename
        self.out_file = out_file
        self.net = net
        self.progress = progress
        self.conditional = _conditional_headers(stored)
        self.served_before = None if stored is None else stored.get("served")
        self.refuse_full = refuse_full
        self.written = 0
        self.validator = None
        self.total = None
        self.resumable = True
        self.attempts = 0
        self.etag = self.last_modified = self.served = self.file_id = None

    def _restart(self):
        _call_local(self.out_file.seek, 0)
        _call_local(self.out_file.truncate)
        self.written = 0

    def _abandon_resume(self, reason):
        """Stop resuming for the rest of the call, drop the copy and fail this attempt."""
        self.resumable = False
        self.validator = None
        self._restart()
        return OSError(
            f"The server answered the download of '{self.url}' with {reason}."
        )

    def _range_length(self, headers):
        """The length of a ``206`` answer that continues the copy at ``written``."""
        content_range = (headers.get("Content-Range") or "").strip()
        match = _CONTENT_RANGE.fullmatch(content_range)
        if match is None or _encoded(headers):
            raise self._abandon_resume(f"the partial content '{content_range}'")
        first, last, total = (int(v) for v in match.groups())
        if (
            first != self.written
            or not first <= last < total
            or (self.total is not None and total != self.total)
        ):
            raise self._abandon_resume(f"the range '{content_range}'")
        self.total = total
        return last - first + 1

    def attempt(self):
        """One request for the file, or for the rest of it after the bytes kept: writes
        what the server sends, reporting the progress, and raises ``OSError`` when the
        answer stops short of its size or holds nothing."""
        resume = self.written > 0 and self.validator is not None
        # The file's own bytes, whatever encodings the caller's headers would accept.
        headers = {"Accept-Encoding": "identity"}
        if resume:
            headers["Range"] = f"bytes={self.written}-"
            headers["If-Range"] = self.validator
        else:
            self._restart()
            headers.update(self.conditional)
        while True:
            if self.refuse_full and not self.conditional:
                raise _Refused()
            self.attempts += 1
            try:
                response = self.net.open(self.url, headers=headers)
                break
            except HTTPError as e:
                if e.code != 304 or not self.conditional:
                    raise
                e.close()
                if e.geturl() == self.served_before:
                    raise _NotModified() from None
                # Not the URL that served the copy: ask for the file itself.
                for name in self.conditional:
                    headers.pop(name)
                self.conditional = {}
        with response:
            if getattr(response, "status", 200) == 206:
                if not resume:
                    raise self._abandon_resume("partial content it was not asked for")
                limit = self._range_length(response.headers)
            else:
                # A full answer, also to a range request whose copy has changed.
                if self.refuse_full:
                    raise _Refused()
                self._restart()
                limit = None
                self.conditional = {}
                self.etag = response.headers.get("ETag")
                self.last_modified = response.headers.get("Last-Modified")
                self.served = getattr(response, "url", None) or self.url
                self.validator = None
                if self.resumable and not _encoded(response.headers):
                    self.validator = _validator(response.headers)
                self.total = _content_length(response.headers.get("Content-Length"))
            _call_local(self.out_file.seek, self.written)
            if self.progress is not None and self.written == 0:
                _call_local(self.progress, 0, self.total)
            received = 0
            while chunk := response.read(_CHUNK):
                received += len(chunk)
                if limit is not None and received > limit:
                    raise self._abandon_resume("more bytes than its range")
                _call_local(self.out_file.write, chunk)
                self.written += len(chunk)
                if self.progress is not None:
                    _call_local(self.progress, self.written, self.total)
        if self.total is not None and self.written != self.total:
            raise OSError(
                f"The download of '{self.url}' stopped after {self.written} of "
                f"{self.total} bytes."
            )
        if self.written == 0:
            raise OSError(f"PBF-file '{self.filename}' from the provider was empty.")


class _LocalWriteError(Exception):
    """Carries an error of the local file or the progress callback out of :func:`_retry`
    and the handlers of network and data failures, without a retry."""


def _call_local(call, *args):
    """Call a method of the local file or the progress callback, marking its error as not
    a network or data failure."""
    try:
        return call(*args)
    except Exception as e:
        raise _LocalWriteError() from e


class UNIT(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4


def convert_unit(size_in_bytes, unit):
    if unit == UNIT.KB:
        return size_in_bytes / 1024
    elif unit == UNIT.MB:
        return size_in_bytes / (1024 * 1024)
    elif unit == UNIT.GB:
        return size_in_bytes / (1024 * 1024 * 1024)
    else:
        return size_in_bytes


def get_file_size(file_name, size_type=UNIT.MB):
    size = Path(file_name).stat().st_size
    return round(convert_unit(size, size_type), 2)


def _new_partial(path):
    """A new, empty file ``<name>.<random>.part`` next to ``path``, with the permissions a
    new file gets."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        partial = path.with_name("%s.%s.part" % (path.name, secrets.token_hex(4)))
        try:
            os.close(os.open(partial, flags, 0o666))
        except FileExistsError:
            continue
        return partial


def write_atomic_path(path, write):
    """Write ``path`` by calling ``write(partial)`` with the path of a new temporary file
    next to it, ``<name>.<random>.part``.

    The temporary file replaces ``path`` only when ``write`` returns, so readers never see
    a partial file, and takes the permissions of the ``path`` it replaces (on POSIX); it is
    removed when ``write`` fails.
    """
    path = Path(path)
    partial = _new_partial(path)
    try:
        write(partial)
        if os.name != "nt":
            try:
                shutil.copymode(path, partial)
            except FileNotFoundError:
                pass
        Path(partial).replace(path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


def write_atomic(path, write):
    """Write ``path`` by calling ``write(file)`` on a new temporary file next to it (see
    :func:`write_atomic_path`), creating its folder when missing."""

    def write_file(partial):
        with open(partial, "wb") as out_file:
            write(out_file)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_atomic_path(path, write_file)


def copy_atomic(source, path):
    """Make ``path`` a copy of the file ``source``: a hard link to it where the file system
    allows one, else a copy of its bytes and permissions. ``path`` is replaced only once the
    copy is complete (see :func:`write_atomic`); nothing is done when it is ``source``
    already.
    """
    path = Path(path)
    if path.exists() and os.path.samefile(source, path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    os.close(fd)
    partial = Path(partial)
    try:
        partial.unlink()
        try:
            os.link(source, partial)
        except OSError:
            shutil.copy(source, partial)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def download_dir():
    """The default directory ``get_data`` downloads PBF extracts into: ``<tempdir>/pyrosm``."""
    return Path(tempfile.gettempdir()) / "pyrosm"


def list_downloads():
    """List the PBF files downloaded by ``get_data`` in the default download directory
    (``<tempdir>/pyrosm``). Returns a sorted list of string paths."""
    directory = download_dir()
    if not directory.is_dir():
        return []
    return sorted(str(p) for p in directory.glob("*.pbf") if p.is_file())


# The suffix `write_atomic` gives the partial file of `<name>`: `.<random>.part`.
_PARTIAL_SUFFIX = re.compile(r"\.[A-Za-z0-9_]+\.part")


def sources_record(path):
    """The file beside ``path`` that records the files it was cropped or merged from."""
    return Path(os.fspath(path) + ".sources")


def validators_record(path):
    """The file beside ``path`` that records the URL, ``ETag`` and ``Last-Modified`` of its
    download, the URL that served it and the file it wrote, which a later ``update`` sends
    back (see :func:`download`)."""
    return Path(os.fspath(path) + ".validators")


def _stored_validators(path, url):
    """The :func:`validators_record` of the file ``path`` when it records a download from
    ``url`` that wrote this very file (see :func:`~pyrosm.utils._file_identity`), else
    None."""
    try:
        stored = json.loads(validators_record(path).read_text(encoding="utf-8"))
        if stored.get("url") == url and stored.get("file") == list(
            _file_identity(path)
        ):
            return stored
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _conditional_headers(stored):
    """The ``If-None-Match`` and ``If-Modified-Since`` headers of the record ``stored``
    (see :func:`_stored_validators`), which ask for the file only if it changed."""
    headers = {}
    if stored and stored.get("etag"):
        headers["If-None-Match"] = stored["etag"]
    if stored and stored.get("last_modified"):
        headers["If-Modified-Since"] = stored["last_modified"]
    return headers


def _record_validators(path, transfer):
    """Record the URL, ``ETag`` and ``Last-Modified`` of ``transfer``, the download of
    ``path``, the URL that served it and the identity of ``path`` (see
    :func:`validators_record`); without either header the record is removed. A record that
    cannot be written, or a ``path`` that is no longer the file ``transfer`` wrote, is left
    out."""
    record = validators_record(path)
    try:
        if not (transfer.etag or transfer.last_modified):
            record.unlink(missing_ok=True)
            return
        identity = _file_identity(path)
        if identity[:2] != transfer.file_id:
            # Another writer replaced the file since.
            return
        stored = {
            "url": transfer.url,
            "served": transfer.served,
            "etag": transfer.etag,
            "last_modified": transfer.last_modified,
            "file": list(identity),
        }
        data = json.dumps(stored).encode("utf-8")
        write_atomic(record, lambda out: out.write(data))
    except OSError:
        pass


def _download_log_dir():
    """The folder of the download logs (see :func:`_download_log`)."""
    return download_dir() / "download-log"


def _download_log(url):
    """The file that records the full downloads of ``url`` in the last 24 hours, which
    :func:`download` counts; deleting it resets the count."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return _download_log_dir() / f"{digest}.json"


def _lock_dir():
    """The folder of the host locks (see :func:`_host_lock`)."""
    return download_dir() / "locks"


@contextlib.contextmanager
def _host_lock(url):
    """Hold the lock of the host of ``url`` while the block runs, so that the downloads
    from one host on this machine go one at a time, whatever their folders; logs a line
    when another download from the host has to finish first."""
    host = urllib.parse.urlsplit(url).hostname or "unknown"
    host = re.sub(r"[^a-z0-9.-]", "_", host)

    def waiting():
        log_event(logger, logging.INFO, "download_waiting", host=host, url=url)

    with exclusive_lock(_lock_dir() / f"{host}.lock", waiting):
        yield


def _logged_time(download):
    """The time of the ``download`` in a download log, in seconds since the epoch."""
    return datetime.fromisoformat(download["time"]).timestamp()


def _recent_downloads(log, now):
    """The full downloads that the download log ``log`` records within 24 hours of ``now``
    (seconds since the epoch), each ``{"time": <ISO 8601>, "bytes": <size>}``; an
    unreadable log records none."""
    try:
        downloads = json.loads(log.read_text(encoding="utf-8"))["downloads"]
        return [d for d in downloads if abs(now - _logged_time(d)) < _DAY]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
        return []


def _log_download(url, log, size, now):
    """Add a full download of ``url`` (``size`` bytes, at ``now``) to its download log
    ``log``, leaving out the downloads older than 24 hours; a log that cannot be written is
    left as it is."""
    downloads = _recent_downloads(log, now)
    time_ = datetime.fromtimestamp(now, timezone.utc).isoformat()
    downloads.append({"time": time_, "bytes": size})
    data = json.dumps({"url": url, "downloads": downloads}, indent=1).encode("utf-8")
    try:
        write_atomic(log, lambda out: out.write(data))
    except OSError:
        pass


def _refusal(url, log, recent):
    """The message of the refusal to download ``url`` in full again after the downloads
    ``recent`` that its download log ``log`` records (see :func:`_recent_downloads`)."""
    times = sorted(_logged_time(d) for d in recent)
    until = datetime.fromtimestamp(
        math.ceil(times[-_FULL_DOWNLOADS_A_DAY] + _DAY), timezone.utc
    )
    return (
        "pyrosm has downloaded '%s' in full %d times in the last 24 hours and refuses to "
        "download it again before %s UTC. The providers serve these files for free and "
        "block clients that download them repeatedly. Reuse the file you have: without "
        "update=True a file that exists is not downloaded again, and with it only a "
        "changed file is. The downloads are recorded in %s; delete it if they were "
        "needed. See %s."
        % (url, len(recent), until.strftime("%Y-%m-%d %H:%M:%S"), log, _RESPONSIBLY)
    )


def _remove_download(path):
    """Remove the file ``path``, the partial files of its interrupted downloads and its
    record (see :func:`sources_record`); returns the number of PBF and partial files
    removed."""
    partials = [
        p
        for p in path.parent.glob(glob.escape(path.name) + ".*.part")
        if _PARTIAL_SUFFIX.fullmatch(p.name[len(path.name) :])
    ]
    removed = [p for p in (path, *partials) if p.is_file()]
    for file in removed:
        file.unlink()
    sources_record(path).unlink(missing_ok=True)
    validators_record(path).unlink(missing_ok=True)
    return len(removed)


def clear_downloads(filepath=None):
    """Remove PBF files downloaded or cropped into the default download directory
    (``<tempdir>/pyrosm``), with the partial files of interrupted downloads and the records
    of the crops. With no ``filepath`` every ``*.pbf`` file there is removed; with a bare
    file name, the file of that name there; with a path, exactly that file. ``filepath``
    must name a ``.osm.pbf`` or ``.osh.pbf`` file. The out-of-core ``cache``, the provider
    indexes and the bundled package data are left untouched. Returns the number of PBF and
    partial files removed."""
    directory = download_dir()
    if filepath is not None:
        name = os.fspath(filepath)
        if not name.lower().endswith((".osm.pbf", ".osh.pbf")):
            raise ValueError(
                "clear_downloads removes .osm.pbf and .osh.pbf files only; got %r."
                % (name,)
            )
        bare = os.path.basename(name) == name
        return _remove_download(directory / name if bare else Path(name))
    if not directory.is_dir():
        return 0
    stems = {p.name for p in directory.glob("*.pbf")}
    stems |= {p.name.rsplit(".", 2)[0] for p in directory.glob("*.pbf.*.part")}
    stems |= {p.name[: -len(".sources")] for p in directory.glob("*.pbf.sources")}
    stems |= {p.name[: -len(".validators")] for p in directory.glob("*.pbf.validators")}
    return sum(_remove_download(directory / stem) for stem in sorted(stems))


def download(
    url,
    filename,
    update,
    target_dir,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
    progress=None,
):
    """Download ``url`` to ``<target_dir>/<filename>`` unless it is there (or ``update``).

    ``target_dir`` defaults to pyrosm's temp directory and is created when missing.
    ``headers``, ``timeout`` and ``opener`` go to every request (see :func:`open_url`). A
    dropped connection is resumed or retried; a failure raises
    :class:`~pyrosm.exceptions.DownloadError`. Returns the file path as a string.

    With ``update``, a file downloaded before is asked for only if it changed: the ``ETag``
    and ``Last-Modified`` its download gave (see :func:`validators_record`) go with the
    request, and the server's 304 answer keeps the file as it is.

    Each full download of ``url`` is recorded in a log in pyrosm's temp directory, wherever
    the file goes. After 3 full downloads of ``url`` in 24 hours, a 4th raises
    :class:`~pyrosm.exceptions.RequestRefusedError` before the server sends the file; a
    file that a 304 answer keeps is not a download. Deleting the log, which the error
    names, resets the count.

    Downloads from one host go one at a time on this machine: while another thread or
    process downloads from the host, the call waits (logging ``download_waiting``), and a
    file the other download wrote meanwhile is not downloaded again unless ``update``.

    ``progress`` shows the download: ``None`` (default) follows
    :func:`pyrosm.enable_logging`, ``True`` draws a bar on stderr (a widget in Jupyter when
    ipywidgets is installed; without a terminal, only a line naming the file), ``False``
    shows nothing, and a callable is called as ``progress(written, total)`` when the server
    starts sending the file and after each chunk of 256 KiB, with the bytes written so far and
    its size, ``None`` when the server does not give it. A resumed download counts on from
    the bytes it kept, a restarted one from 0.
    """
    validate_progress(progress)
    target_dir = download_dir() if target_dir is None else Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir.resolve() / Path(filename).name
    net = _Net(headers, timeout, opener)

    if update or not filepath.exists():
        with _host_lock(url):
            # Another download may have written the file while this one waited.
            if update or not filepath.exists():
                _fetch_into(filepath, url, filename, net, progress)
    return str(filepath)


def _fetch_into(filepath, url, filename, net, progress):
    """Download ``url`` into ``filepath`` for :func:`download`: conditionally when the
    record of an earlier download of it is there, refusing a full download that the
    download log does not allow, and logging and recording what was done."""
    stored = _stored_validators(filepath, url) if filepath.exists() else None
    log = _download_log(url)
    recent = _recent_downloads(log, time.time())
    refuse = len(recent) >= _FULL_DOWNLOADS_A_DAY
    transfers = []

    def fetch(out_file):
        transfer = _Transfer(url, filename, out_file, net, report, stored, refuse)
        transfers.append(transfer)
        while True:
            kept = transfer.written
            try:
                _retry(transfer.attempt)
                # The device and inode of the file written, which its rename keeps.
                transfer.file_id = _call_local(_file_identity, out_file)[:2]
                _log_download(url, log, transfer.written, time.time())
                return
            except _FETCH_ERRORS as e:
                # A resumable round that kept more of the file earns another round.
                gained = transfer.written > kept
                if _retryable(e) and transfer.validator is not None and gained:
                    continue
                n = transfer.attempts
                raise DownloadError(
                    f"Could not download '{url}' ({n} attempt"
                    f"{'' if n == 1 else 's'}): {e}",
                    url=url,
                    status=e.code if isinstance(e, HTTPError) else None,
                    attempts=n,
                ) from e

    # write_atomic moves the file into place only when complete, so a failed download
    # never leaves a partial file that a later call would reuse. Errors creating or
    # replacing the local file propagate as they are.
    start = time.perf_counter()
    while True:
        try:
            with reporting(progress, "Downloading %s" % filepath.name) as report:
                try:
                    write_atomic(filepath, fetch)
                except _LocalWriteError as e:
                    raise e.__cause__
            break
        except _NotModified:
            if _stored_validators(filepath, url) == stored:
                log_event(
                    logger,
                    logging.INFO,
                    "download_unchanged",
                    file=filepath.name,
                    seconds=time.perf_counter() - start,
                    path=filepath,
                )
                return
            # The file changed while the server was asked: download it in full.
            stored = None
        except _Refused:
            sent = sum(transfer.attempts for transfer in transfers)
            raise RequestRefusedError(
                _refusal(url, log, recent), url=url, attempts=sent
            ) from None
    _record_validators(filepath, transfers[-1])

    log_event(
        logger,
        logging.INFO,
        "download",
        file=filepath.name,
        megabytes=get_file_size(filepath),
        seconds=time.perf_counter() - start,
        path=filepath,
    )
