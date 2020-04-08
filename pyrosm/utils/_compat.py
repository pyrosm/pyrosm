from shapely.geos import geos_version_string as shapely_geos_version
from pygeos import geos_capi_version_string
import warnings

# shapely has something like: "3.6.2-CAPI-1.10.2 4d2925d6"
# pygeos has something like: "3.6.2-CAPI-1.10.2"
if not shapely_geos_version.startswith(geos_capi_version_string):
    warnings.warn(
        "The Shapely GEOS version ({}) is incompatible with the GEOS "
        "version PyGEOS was compiled with ({}). The tool will work "
        "but it runs a bit slower.".format(
            shapely_geos_version, geos_capi_version_string
        )
    )
    PYGEOS_SHAPELY_COMPAT = False
else:
    PYGEOS_SHAPELY_COMPAT = True
