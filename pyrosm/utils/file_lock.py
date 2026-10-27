import contextlib
import errno
import os
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

# The errors with which a lock that another open file holds is refused.
_HELD = (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, errno.EDEADLK)
# How long a lock on Windows, which cannot wait for one, waits between tries.
_POLL_SECONDS = 0.1
# The descriptors of the lock files this process has open, and the locks it holds, each as
# its path and the thread holding it.
_open_fds = set()
_holders = set()


def _let_go_in_child():
    """In a child process made by fork, point the descriptors of the lock files the parent
    has open at the null device, so that the child does not keep their locks."""
    for fd in list(_open_fds):
        with contextlib.suppress(OSError):
            null = os.open(os.devnull, os.O_RDWR)
            try:
                os.dup2(null, fd, inheritable=False)
            finally:
                os.close(null)
    _open_fds.clear()
    _holders.clear()


with contextlib.suppress(AttributeError):  # no fork on Windows
    os.register_at_fork(after_in_child=_let_go_in_child)


def _try_lock(f):
    """Take the exclusive lock of the open file ``f`` if no other open file holds it;
    returns whether it did."""
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as e:
        if e.errno in _HELD:
            return False
        raise
    return True


def _lock(f, waiting):
    """Take the exclusive lock of the open file ``f``, calling ``waiting()`` and waiting
    for it when another open file holds it; returns False when the file system cannot
    lock."""
    try:
        if _try_lock(f):
            return True
    except OSError:
        return False
    waiting()
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        else:
            while not _try_lock(f):
                time.sleep(_POLL_SECONDS)
    except OSError:
        return False
    return True


def _unlock(f):
    """Release the lock that :func:`_lock` took on the open file ``f``."""
    if fcntl is not None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    else:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def exclusive_lock(path, waiting):
    """Hold the operating system's exclusive lock on the file ``path`` (created with its
    folder when missing) while the block runs, waiting for it while another open file
    holds it, in this or another process; ``waiting()`` is called once before the wait.
    The system releases the lock of a process that ends, and a child made by fork does
    not keep it. A thread that holds the lock already gets a ``RuntimeError`` instead of
    waiting for itself. Where the file cannot be opened or locked, the block runs without
    the lock."""
    key = (os.path.abspath(path), threading.get_ident())
    if key in _holders:
        raise RuntimeError(
            "This thread holds the lock %s already: a nested call, such as from a "
            "callback, would wait for itself." % path
        )
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "a+b")
    except OSError:
        f = None
    if f is None:
        yield
        return
    with f:
        fd, locked = f.fileno(), False
        try:
            _open_fds.add(fd)
            locked = _lock(f, waiting)
            if locked:
                _holders.add(key)
            yield
        finally:
            if locked:
                _holders.discard(key)
                with contextlib.suppress(OSError):
                    _unlock(f)
            _open_fds.discard(fd)
