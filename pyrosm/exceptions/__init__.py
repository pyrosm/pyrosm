class PBFException(Exception):
    pass


class PBFNotImplemented(PBFException):
    pass


class InvalidOSMFileError(PBFException):
    pass


class ExtractDownloadError(Exception):
    """No extract that contains the requested area could be downloaded.

    ``failed`` lists ``(url, error message)`` for every extract that was tried.
    """

    def __init__(self, message, failed):
        super().__init__(message)
        self.failed = failed
