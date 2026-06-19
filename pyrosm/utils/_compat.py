import warnings

# python-igraph is an optional dependency
try:
    import igraph

    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

# networkx is an optional dependency
try:
    import networkx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# pandana is an optional dependency
try:
    import pandana

    HAS_PANDANA = True
except ImportError:
    HAS_PANDANA = False

# pandarm is an optional dependency (the maintained, NumPy 2-compatible fork of pandana)
try:
    import pandarm

    HAS_PANDARM = True
except ImportError:
    HAS_PANDARM = False

# pyarrow is an optional dependency, needed only to write GeoParquet via output=
try:
    import pyarrow.parquet  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def require_pyarrow():
    """Raise an actionable ImportError when an output= GeoParquet write is requested
    without the optional pyarrow dependency installed."""
    if not HAS_PYARROW:
        raise ImportError(
            "Writing to GeoParquet (output=...) requires the optional 'pyarrow' "
            "dependency. Install it with `pip install pyarrow`."
        )
