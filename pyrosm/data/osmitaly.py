import json
import urllib.request
URL_wmi = "https://osmit-estratti.wmcloud.org/dati/poly/"
suffix_wmi = "_poly.osm.pbf"
italian_regions_url = "regioni/pbf/"
italian_provinces_url = "province/pbf/"
#italian_municipalities_url = "comuni/pbf/"
italian_regions_json_url = "https://raw.githubusercontent.com/GISdevio/estratti_OSM_Italia/main/webapp/public/static/boundaries/limits_IT_regions.json"
italian_provinces_json_url = "https://raw.githubusercontent.com/GISdevio/estratti_OSM_Italia/main/webapp/public/static/boundaries/limits_IT_provinces.json"

## read the data for the regions
r = urllib.request.urlopen(italian_regions_json_url)
data_json = r.read()
encoding = r.info().get_content_charset('utf-8')
json_italian_regions = json.loads(data_json.decode(encoding))
data_italian_regions = json_italian_regions['objects']['limits_IT_regions']['geometries']
italian_resources = {}
italian_regions = []
for r in data_italian_regions:
    name = r['properties']['name'].replace("/","-")
    source = r['properties']['name'].lower().replace(" ","_").replace("'","_").replace("/","-")
    istat = r['properties']['istat']
    filename = istat + "_" + name
    italian_resources[source] = filename
    italian_regions.append(source)

## read the data for the provinces
r = urllib.request.urlopen(italian_provinces_json_url)
data_json = r.read()
encoding = r.info().get_content_charset('utf-8')
json_italian_provinces = json.loads(data_json.decode(encoding))
data_italian_provinces = json_italian_provinces['objects']['limits_IT_provinces']['geometries']
italian_provinces = []
for r in data_italian_provinces:
    name = r['properties']['name'].replace("/","-")
    source = "provincia di " + r['properties']['name'].lower().replace(" ","_").replace("'","_").replace("/","-")
    istat = r['properties']['istat']
    filename = istat + "_" + name
    italian_resources[source] = filename
    italian_provinces.append(source)

class ItalianProvinces:
    available = italian_provinces


    # Create data sources
    _sources = {
        region: {
            "name": region + suffix_wmi,
            "url": URL_wmi + italian_provinces_url + italian_resources[region] + suffix_wmi,
        }
        for region in italian_provinces
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class ItalianRegions:
    #regions = italian_regions
    italian_provinces = ItalianProvinces()

    available = (
        italian_regions
        + italian_provinces.available
    )
    available.sort()

    # Create data sources
    _sources = {
        region: {
            "name": region + suffix_wmi,
            "url": URL_wmi + italian_regions_url + italian_resources[region] + suffix_wmi,
        }
        for region in italian_regions
    }

    for region in italian_provinces.available:
        _sources[region] = {
            "name": region + suffix_wmi,
            "url": URL_wmi + italian_provinces_url + italian_resources[region] + suffix_wmi,
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available
