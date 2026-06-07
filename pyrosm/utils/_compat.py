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
