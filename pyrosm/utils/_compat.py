"""Whether the optional graph libraries are installed, found without importing them:
python-igraph, networkx, pandana and pandarm (the maintained, NumPy 2-compatible fork of
pandana). Each exporter imports its library when it is called."""

from importlib.util import find_spec

HAS_IGRAPH = find_spec("igraph") is not None
HAS_NETWORKX = find_spec("networkx") is not None
HAS_PANDANA = find_spec("pandana") is not None
HAS_PANDARM = find_spec("pandarm") is not None
