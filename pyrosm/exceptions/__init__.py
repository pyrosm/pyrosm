"""Exceptions raised by pyrosm.

- :class:`PBFException`: the base class of the two errors about a PBF file's content below.
- :class:`InvalidOSMFileError`: a file is not a readable OSM PBF file (a
  :class:`PBFException`).
- :class:`PBFNotImplemented`: a PBF file requires a feature pyrosm does not support (a
  :class:`PBFException`).
- :class:`ExtractNotFoundError`: no provider extract contains the requested area (a
  ``ValueError``); raised by :func:`pyrosm.get_data_by_area`.
- :class:`ExtractDownloadError`: none of the extracts that contain the area could be
  downloaded; raised by :func:`pyrosm.get_data_by_area`.
- :class:`DownloadError`: one file could not be downloaded, after the retries (an
  :class:`ExtractDownloadError`, a ``ValueError`` and an ``OSError``); raised by
  :func:`pyrosm.get_data` and the other download functions.
"""


class PBFException(Exception):
    """Base class of the errors about the content of a PBF file."""


class PBFNotImplemented(PBFException):
    """The PBF file requires a feature pyrosm does not support, such as node locations
    stored on ways; the message names the file and the feature."""


class InvalidOSMFileError(PBFException):
    """The file is not a readable OSM PBF file, for example because it is truncated or
    another format; the message names the file."""


class ExtractNotFoundError(ValueError):
    """No Geofabrik, BBBike or Movisda extract contains the requested area.

    A ``ValueError``, so code that catches ``ValueError`` keeps working. An invalid area
    (empty, or without width or height) raises a plain ``ValueError`` instead.
    """


class ExtractDownloadError(Exception):
    """No extract that contains the requested area could be downloaded.

    Attributes
    ----------
    failed : list of (str, str)
        ``(url, error message)`` for every extract that was tried.
    errors : list of DownloadError
        The :class:`DownloadError` of every extract that was tried, in the order of ``failed``.
    url, status, attempts
        Set on a :class:`DownloadError` for its one URL: the URL, the HTTP status of the last
        attempt (``None`` for a network error) and the number of attempts; ``None`` here.
    """

    def __init__(
        self, message, failed=(), url=None, status=None, attempts=None, errors=()
    ):
        super().__init__(message)
        self.failed = list(failed)
        self.errors = list(errors)
        self.url = url
        self.status = status
        self.attempts = attempts


class DownloadError(ExtractDownloadError, ValueError, OSError):
    """A file could not be downloaded, after retrying network errors and HTTP status 408, 425,
    429 and 5xx.

    ``url``, ``status`` (the HTTP status of the last attempt, ``None`` for a network error) and
    ``attempts`` describe the failure; the last error is its ``__cause__``. It is also a
    ``ValueError`` and an ``OSError``, the types these failures had before.
    """
