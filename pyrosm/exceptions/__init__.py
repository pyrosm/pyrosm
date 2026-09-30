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

    ``failed`` lists ``(url, error message)`` for every extract that was tried.
    """

    def __init__(self, message, failed):
        super().__init__(message)
        self.failed = failed
