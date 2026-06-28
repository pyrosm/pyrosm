"""Generate the interactive New York buildings map embedded on the landing page.

Reads buildings for a Lower/Midtown Manhattan bounding box from the New York City extract,
colours them by construction year, and writes a self-contained lonboard map.
"""
from pathlib import Path

import matplotlib as mpl
from lonboard import Map, PolygonLayer
from lonboard.basemap import CartoStyle, MaplibreBasemap
from lonboard.colormap import apply_continuous_cmap

from pyrosm import OSM, get_data

OUT = Path(__file__).parent / "_static" / "ny_buildings.html"
# Lower Manhattan to Midtown, kept on the island (min_lon, min_lat, max_lon, max_lat)
BBOX = [-74.017, 40.703, -73.972, 40.762]


def extract_year(series):
    """Pull a 4-digit year out of a free-text OSM column, as float (NaN if none)."""
    return series.astype(str).str.extract(r"(\d{4})")[0].astype(float)


def join_address(row):
    parts = [row.get("addr:street"), row.get("addr:housenumber")]
    parts = [p for p in parts if isinstance(p, str) and p]
    return " ".join(parts) if parts else None


def main():
    osm = OSM(get_data("New York City"), bounding_box=BBOX)
    gdf = osm.get_buildings(extra_attributes=["start_date", "year_of_construction"])
    gdf = gdf.loc[gdf.geometry.geom_type != "MultiLineString"].copy()

    gdf["year"] = extract_year(gdf["year_of_construction"]).combine_first(
        extract_year(gdf["start_date"])
    )
    gdf["address"] = gdf.apply(join_address, axis=1)

    attr_cols = [c for c in ["year", "name", "address", "building", "amenity"] if c in gdf.columns]
    keep = attr_cols + [gdf.geometry.name]

    dated = gdf[gdf["year"].notna()].copy()
    undated = gdf[gdf["year"].isna()].copy()

    years = dated["year"].to_numpy()
    norm = mpl.colors.Normalize(vmin=years.min(), vmax=years.max(), clip=True)
    colors = apply_continuous_cmap(norm(years), mpl.colormaps["viridis"], alpha=0.85)

    dated_layer = PolygonLayer.from_geopandas(
        dated[keep],
        get_fill_color=colors,
        get_line_color=[40, 40, 40, 100],
        pickable=True,
        auto_highlight=True,
    )
    undated_layer = PolygonLayer.from_geopandas(
        undated[keep],
        get_fill_color=[180, 180, 180, 120],
        get_line_color=[120, 120, 120, 100],
        pickable=True,
        auto_highlight=True,
    )

    m = Map(
        [undated_layer, dated_layer],
        basemap=MaplibreBasemap(style=CartoStyle.DarkMatter),
        view_state={"longitude": -73.9945, "latitude": 40.7325, "zoom": 12.4},
    )
    m.to_html(OUT, title="New York City buildings by construction year (pyrosm)")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) | "
          f"dated={len(dated)} undated={len(undated)} total={len(gdf)}")


if __name__ == "__main__":
    main()
