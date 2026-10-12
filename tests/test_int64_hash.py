import numpy as np
import pytest


def test_int64_hash_tables():
    """Int64Set and Int64Map answer membership and lookups for int64 ids, also from
    read-only arrays and lists; a key given twice maps to its last value, and more keys
    than one table holds raise a MemoryError before the input is copied."""
    from pyrosm.int64_hash import Int64Map, Int64Set

    ids = np.array([5, -3, 2**62, 5], dtype=np.int64)
    ids.flags.writeable = False

    ids_set = Int64Set(ids)
    assert len(ids_set) == 3
    assert 2**62 in ids_set and 4 not in ids_set
    assert ids_set.isin([5, 4, -3]).tolist() == [True, False, True]
    assert ids_set.contains_any([7, -3])
    assert not ids_set.contains_any([7])
    assert not ids_set.contains_any(None)
    assert len(Int64Set()) == 0
    assert Int64Set().isin([1]).tolist() == [False]

    rows = Int64Map(ids, np.arange(4))
    assert len(rows) == 3
    assert 5 in rows and 4 not in rows
    assert rows.get_many([5, 4, 2**62], -1).tolist() == [3, -1, 2]
    assert len(Int64Map()) == 0
    with pytest.raises(ValueError):
        Int64Map([1, 2], [1])

    too_many = np.broadcast_to(np.int64(0), (2**31,))
    with pytest.raises(MemoryError):
        Int64Set(too_many)
    with pytest.raises(MemoryError):
        Int64Map(too_many, too_many)
