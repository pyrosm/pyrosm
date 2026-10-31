"""Geofabrik's extract index, and :func:`get_data_by_bbox`, which downloads (and by default
crops) the OSM data for a bounding box as :func:`pyrosm.get_data_by_area` does.

The index is a vendored snapshot of Geofabrik's ``index-v1.json``
(``geofabrik_index.geojson.gz``), a GeoJSON ``FeatureCollection`` of every extract's extent
polygon and PBF URL. Refresh the snapshot with ``scripts/update_extract_indexes.py``.
"""

import collections
import email.utils
import functools
import gzip
import json
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import geopandas as gpd
from shapely.geometry import box

from pyrosm.utils import validate_bounding_box
from pyrosm.utils.download import _TIMEOUT

_INDEX_PATH = Path(__file__).parent / "geofabrik_index.geojson.gz"
_INDEX_URL = "https://download.geofabrik.de/index-v1.json"

# Rank covering extents by true size in an equal-area projection; lon/lat degree
# area over-states size near the poles and can mis-rank nested extracts.
_EQUAL_AREA_CRS = "EPSG:6933"

_index_cache = None


def _features_to_gdf(features):
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    # The live index nests download links under "urls"; the vendored snapshot
    # pre-flattens to a "pbf" column. Normalise both to a "pbf" column.
    if "pbf" not in gdf.columns and "urls" in gdf.columns:
        gdf["pbf"] = gdf["urls"].apply(
            lambda u: u.get("pbf") if isinstance(u, dict) else None
        )
    return gdf


@functools.lru_cache(maxsize=None)
def _shared_file_names():
    """The file names, in lower case, that several extracts of the vendored index have,
    such as ``georgia-latest.osm.pbf`` of the US state and of the country."""
    with gzip.open(_INDEX_PATH, "rt", encoding="utf-8") as f:
        features = json.load(f)["features"]
    counts = collections.Counter(
        PurePosixPath(urlparse(url).path).name.lower()
        for url in (feature["properties"].get("pbf") for feature in features)
        if url
    )
    return frozenset(name for name, count in counts.items() if count > 1)


def extract_file_name(url, name=None):
    """The local file name of the extract at ``url``: ``name`` (by default the file name
    of the URL), or, when other Geofabrik extracts have the same file name, the URL's
    file name with its region in front (``us-georgia-latest.osm.pbf``,
    ``europe-georgia-latest.osm.pbf``)."""
    path = PurePosixPath(urlparse(url).path)
    if path.name.lower() in _shared_file_names():
        return "%s-%s" % (path.parent.name, path.name)
    return name or path.name


def _load_index():
    """pyrosm's copy of Geofabrik's index (read once), with the day Geofabrik published it
    as ``attrs["snapshot_date"]``."""
    global _index_cache
    if _index_cache is None:
        with gzip.open(_INDEX_PATH, "rt", encoding="utf-8") as f:
            collection = json.load(f)
        _index_cache = _features_to_gdf(collection["features"])
        published = email.utils.parsedate_to_datetime(
            collection["geofabrik_snapshot_date"]
        )
        _index_cache.attrs["snapshot_date"] = published.date().isoformat()
    return _index_cache


def _bbox_to_polygon(bbox):
    """The box of ``bbox`` (any form :func:`~pyrosm.utils.validate_bounding_box` takes; a
    geometry's envelope) as a Polygon."""
    checked = validate_bounding_box(bbox, "bbox")
    return box(*(checked if isinstance(checked, list) else checked.bounds))


def _fmt_coord(value):
    return ("%.5f" % value).rstrip("0").rstrip(".")


def _bbox_filename(bounds):
    return "bbox_%s_%s_%s_%s.osm.pbf" % tuple(_fmt_coord(v) for v in bounds)


def _default_target_dir(directory):
    if directory is not None:
        return Path(directory)
    return Path(tempfile.gettempdir()) / "pyrosm"


def get_data_by_bbox(
    bbox,
    crop=True,
    download=True,
    update=False,
    directory=None,
    output_path=None,
    progress=None,
    headers=None,
    timeout=_TIMEOUT,
    opener=None,
    strategy="single",
    must_cover=None,
    workers=None,
    tolerance=0.001,
):
    """Download (and by default crop) the OSM data covering a bounding box.

    Gets the data for the box as :func:`pyrosm.get_data_by_area` does: the smallest
    Geofabrik, BBBike or Movisda extract (or set of extracts) that covers it, downloaded
    and -- by default -- cropped to the box.

    See https://pyrosm.readthedocs.io/en/stable/download_responsibly.html
    for what the data providers ask and how pyrosm keeps to it.

    Parameters
    ----------
    bbox : list | tuple | numpy.ndarray | shapely geometry | GeoDataFrame | GeoSeries
        The area of interest as ``[minx, miny, maxx, maxy]`` in lon/lat, a Shapely
        geometry (its bounding box is used), or a GeoDataFrame/GeoSeries (its total
        bounds are used).

    crop : bool
        When ``True`` (default), crop the downloaded extract to ``bbox`` and return
        the cropped file, named ``bbox_<minx>_<miny>_<maxx>_<maxy>.osm.pbf`` and used
        again while it and its extract are unchanged. When ``False``, return the full
        downloaded extract.

    download : bool
        When ``True`` (default), download the covering extract. When ``False``,
        skip the download and return the PBF URL of the extract it would download
        instead (a list of URLs for a set of extracts).

    update, directory
        As for :func:`pyrosm.get_data_by_area`.

    output_path : str, optional
        Path for the file pyrosm writes, as for :func:`pyrosm.get_data_by_area`.
        Ignored when ``download=False``.

    progress : bool | callable | None
        ``None`` (default) follows :func:`pyrosm.enable_logging`, which is ``True`` unless
        set. ``True`` shows the download as a bar on stderr, a widget in Jupyter when
        ipywidgets is installed (``pip install "pyrosm[notebook]"``); without a terminal,
        only a line naming the file is printed. ``False`` shows nothing. A callable is called
        as ``progress(written, total)`` after each chunk, with the bytes written so far and
        the file size, ``None`` when the server does not give it.
        The crop shows its progress the same way; a callable is then called with the
        percentage of the crop done.

    headers, timeout, opener, strategy, must_cover, workers, tolerance
        As for :func:`pyrosm.get_data_by_area`.

    Returns
    -------
    str | list
        The cropped file path (default), the full extract path (``crop=False``), or
        the covering extract's PBF URL (``download=False``; a list of URLs for a set of
        extracts).

    Raises
    ------
    ValueError
        If ``bbox`` is not a usable box, or no extract covers it
        (:class:`~pyrosm.exceptions.ExtractNotFoundError`).
    """
    from pyrosm.data.extract_index import _get_area, _url_result

    result = _get_area(
        _bbox_to_polygon(bbox),
        crop,
        update,
        directory,
        output_path,
        headers,
        timeout,
        opener,
        strategy,
        must_cover,
        workers,
        progress,
        download=download,
        tolerance=tolerance,
    )
    return _url_result(result) if not download else result.path
