.. _reference:

API reference
=============

The :class:`~pyrosm.OSM` class is the main entry point: construct it once from a PBF file,
then call a feature method to read a layer into a GeoDataFrame, export a network to a graph,
or write data back to PBF. Module-level helpers download PBF extracts and simplify graphs.

OSM reader
----------

.. currentmodule:: pyrosm

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   OSM

Reading OSM features
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   OSM.get_network
   OSM.get_buildings
   OSM.get_pois
   OSM.get_landuse
   OSM.get_natural
   OSM.get_boundaries
   OSM.get_data_by_custom_criteria

Exporting to a graph
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   OSM.to_graph

Saving, cropping and merging PBF files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   OSM.to_pbf
   OSM.write_pbf
   merge_pbf

Cache and downloads
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   OSM.list_cache
   OSM.list_downloads
   OSM.clear_cache
   OSM.clear_downloads

Downloading data
----------------

.. autosummary::
   :toctree: api/

   get_data
   find_extracts
   get_data_by_area
   get_data_by_bbox
   geocode
   get_data_by_geocoding

Exceptions
----------

.. currentmodule:: pyrosm.exceptions

.. autosummary::
   :toctree: api/

   ExtractNotFoundError
   ExtractDownloadError
   DownloadError
   InvalidOSMFileError
   PBFNotImplemented
   PBFException

Graph simplification
--------------------

.. currentmodule:: pyrosm.graph_simplify

.. autosummary::
   :toctree: api/

   simplify_graph
