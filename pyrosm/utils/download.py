import email.utils
import enum
import http.client
import logging
import re
import ssl
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import certifi

from pyrosm import __version__
from pyrosm.exceptions import DownloadError

USER_AGENT = "pyrosm/%s (+https://github.com/pyrosm/pyrosm)" % __version__

logger = logging.getLogger(__name__)

_FETCH_ERRORS = (OSError, http.client.HTTPException)
_RETRY_STATUSES = (408, 425, 429)
_TIMEOUT = 60
_ATTEMPTS = 3
_BACKOFF = 1.0
_MAX_RETRY_AFTER = 60
_CHUNK = 1 << 20
# Headers that belong to the URL they were given for and are not sent on after a redirect.
_ORIGIN_BOUND = ("authorization", "cookie", "proxy-authorization", "host")
_STRONG_ETAG = re.compile(r'"[\x21\x23-\x7e\x80-\xff]*"')
_CONTENT_RANGE = re.compile(r"bytes (\d{1,19})-(\d{1,19})/(\d{1,19})")
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


class _Transfer:
    """One file downloaded into ``out_file``, over as many attempts as it takes.

    An attempt after a dropped connection asks for the rest of the file with ``Range`` and
    ``If-Range`` when the server gave a strong validator, and starts over otherwise. After a
    partial answer that does not fit the copy, the rest of the call starts over every time.
    """

    def __init__(self, url, filename, out_file, net):
        self.url = url
        self.filename = filename
        self.out_file = out_file
        self.net = net
        self.written = 0
        self.validator = None
        self.total = None
        self.resumable = True
        self.attempts = 0

    def _restart(self):
        _local(self.out_file.seek, 0)
        _local(self.out_file.truncate)
        self.written = 0

    def _mismatch(self, reason):
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
            raise self._mismatch(f"the partial content '{content_range}'")
        first, last, total = (int(v) for v in match.groups())
        if (
            first != self.written
            or not first <= last < total
            or (self.total is not None and total != self.total)
        ):
            raise self._mismatch(f"the range '{content_range}'")
        self.total = total
        return last - first + 1

    def attempt(self):
        self.attempts += 1
        resume = self.written > 0 and self.validator is not None
        # The file's own bytes, whatever encodings the caller's headers would accept.
        headers = {"Accept-Encoding": "identity"}
        if resume:
            headers["Range"] = f"bytes={self.written}-"
            headers["If-Range"] = self.validator
        else:
            self._restart()
        with self.net.open(self.url, headers=headers) as response:
            if getattr(response, "status", 200) == 206:
                if not resume:
                    raise self._mismatch("partial content it was not asked for")
                limit = self._range_length(response.headers)
            else:
                # A full answer, also to a range request whose copy has changed.
                self._restart()
                limit = None
                self.validator = None
                if self.resumable and not _encoded(response.headers):
                    self.validator = _validator(response.headers)
                self.total = _content_length(response.headers.get("Content-Length"))
            _local(self.out_file.seek, self.written)
            received = 0
            while chunk := response.read(_CHUNK):
                received += len(chunk)
                if limit is not None and received > limit:
                    raise self._mismatch("more bytes than its range")
                _local(self.out_file.write, chunk)
                self.written += len(chunk)
        if self.total is not None and self.written != self.total:
            raise OSError(
                f"The download of '{self.url}' stopped after {self.written} of "
                f"{self.total} bytes."
            )
        if self.written == 0:
            raise OSError(f"PBF-file '{self.filename}' from the provider was empty.")


class _LocalWriteError(Exception):
    """Carries an ``OSError`` of the local file out of :func:`_retry` without a retry."""


def _local(call, *args):
    """Call a method of the local file, marking its ``OSError`` as not a network failure."""
    try:
        return call(*args)
    except OSError as e:
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


def write_atomic(path, write):
    """Write ``path`` by calling ``write(file)`` on a new temporary file next to it.

    The temporary file gets a unique name and replaces ``path`` only when ``write`` returns, so
    readers never see a partial file; it is removed when ``write`` fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with open(fd, "wb") as out_file:
            write(out_file)
        Path(partial).replace(path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
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


def clear_downloads(filepath=None):
    """Remove PBF files downloaded by ``get_data`` from the default download directory
    (``<tempdir>/pyrosm``). With no ``filepath`` every downloaded ``*.pbf`` there is removed; with
    a ``filepath`` (a path or a bare filename) only that file is removed. The out-of-core ``cache``
    subdirectory and the bundled package data are left untouched. Returns the number of files
    removed."""
    directory = download_dir()
    if not directory.is_dir():
        return 0
    if filepath is None:
        targets = sorted(p for p in directory.glob("*.pbf") if p.is_file())
    else:
        target = directory / Path(filepath).name
        targets = [target] if target.is_file() else []
    removed = 0
    for path in targets:
        path.unlink()
        removed += 1
    return removed


def download(
    url, filename, update, target_dir, headers=None, timeout=_TIMEOUT, opener=None
):
    """Download ``url`` to ``<target_dir>/<filename>`` unless it is there (or ``update``).

    ``target_dir`` defaults to pyrosm's temp directory and is created when missing.
    ``headers``, ``timeout`` and ``opener`` go to every request (see :func:`open_url`). A
    dropped connection is resumed or retried; a failure raises
    :class:`~pyrosm.exceptions.DownloadError`. Returns the file path as a string.
    """
    target_dir = download_dir() if target_dir is None else Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir.resolve() / Path(filename).name
    net = _Net(headers, timeout, opener)

    # Check if file exists
    file_exists = False
    if filepath.exists():
        file_exists = True

    # Download data to temp if it does not exist or if update is requested
    if update or file_exists is False:

        def fetch(out_file):
            transfer = _Transfer(url, filename, out_file, net)
            while True:
                kept = transfer.written
                try:
                    return _retry(transfer.attempt)
                except _FETCH_ERRORS as e:
                    # A resumable round that kept more of the file earns another round.
                    progress = transfer.written > kept
                    if _retryable(e) and transfer.validator is not None and progress:
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
        try:
            write_atomic(filepath, fetch)
        except _LocalWriteError as e:
            raise e.__cause__

        logger.info(
            "Downloaded Protobuf data '%s' (%s MB) to '%s'",
            filepath.name,
            get_file_size(filepath),
            filepath,
        )
    return str(filepath)
