from shapely.geos import geos_version_string as shapely_geos_version
from pygeos import geos_capi_version_string
import warnings

# shapely has something like: "3.6.2-CAPI-1.10.2 4d2925d6"
# pygeos has something like: "3.6.2-CAPI-1.10.2"
if not shapely_geos_version.startswith(geos_capi_version_string):
    PYGEOS_SHAPELY_COMPAT = False
else:
    PYGEOS_SHAPELY_COMPAT = True

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
