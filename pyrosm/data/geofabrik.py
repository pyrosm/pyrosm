# BASE
URL = "http://download.geofabrik.de/"
suffix = "-latest.osm.pbf"
africa_url = "africa/"
asia_url = "asia/"
australia_oceania_url = "australia-oceania/"
central_america_url = "central-america/"
europe_url = "europe/"
north_america_url = "north-america/"
south_america_url = "south-america/"

baden_wuerttemberg_url = "europe/germany/baden-wuerttemberg/"
bayern_url = "europe/germany/bayern/"
brazil_url = "south-america/brazil/"
canada_url = "north-america/canada/"
england_url = "europe/great-britain/england/"
france_url = "europe/france/"
gb_url = "europe/great-britain/"
germany_url = "europe/germany/"
italy_url = "europe/italy/"
japan_url = "asia/japan/"
netherlands_url = "europe/netherlands/"
nordrhein_wesfalen_url = "europe/germany/nordrhein-westfalen/"
poland_url = "europe/poland/"
russia_url = "russia/"
usa_url = "north-america/us/"


class USA:
    # State level data sources
    regions = [
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "colorado",
        "connecticut",
        "delaware",
        "district_of_columbia",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "new_hampshire",
        "new_mexico",
        "new_york",
        "new_jersey",
        "north_carolina",
        "north_dakota",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "puerto_rico",
        "rhode_island",
        "south_carolina",
        "south_dakota",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "west_virginia",
        "wisconsin",
        "wyoming",
    ]

    available = regions + ["southern_california", "northern_california"]
    available.sort()

    country = {"name": "us" + suffix, "url": URL + north_america_url + "us" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + usa_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    # Add California separately
    _sources["southern_california"] = {
        "name": "socal" + suffix,
        "url": URL + "north-america/us/california/socal-latest.osm.pbf",
    }
    _sources["northern_california"] = {
        "name": "norcal" + suffix,
        "url": URL + "north-america/us/california/norcal-latest.osm.pbf",
    }
    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class France:
    regions = [
        "alsace",
        "aquitaine",
        "auvergne",
        "basse_normandie",
        "bourgogne",
        "bretagne",
        "centre",
        "champagne_ardenne",
        "corse",
        "franche_comte",
        "guadeloupe",
        "guyane",
        "haute_normandie",
        "ile_de_france",
        "languedoc_roussillon",
        "limousin",
        "lorraine",
        "martinique",
        "mayotte",
        "midi_pyrenees",
        "nord_pas_de_calais",
        "pays_de_la_loire",
        "picardie",
        "poitou_charentes",
        "provence_alpes_cote_d_azur",
        "reunion",
        "rhone_alpes",
    ]

    available = regions
    available.sort()

    country = {"name": "france" + suffix, "url": URL + europe_url + "france" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + france_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class England:
    regions = [
        "bedfordshire",
        "berkshire",
        "bristol",
        "buckinghamshire",
        "cambridgeshire",
        "cheshire",
        "cornwall",
        "cumbria",
        "derbyshire",
        "devon",
        "dorset",
        "durham",
        "east_sussex",
        "east_yorkshire_with_hull",
        "essex",
        "gloucestershire",
        "greater_london",
        "greater_manchester",
        "hampshire",
        "herefordshire",
        "hertfordshire",
        "isle_of_wight",
        "kent",
        "lancashire",
        "leicestershire",
        "lincolnshire",
        "merseyside",
        "norfolk",
        "north_yorkshire",
        "northamptonshire",
        "northumberland",
        "nottinghamshire",
        "oxfordshire",
        "rutland",
        "shropshire",
        "somerset",
        "south_yorkshire",
        "staffordshire",
        "suffolk",
        "surrey",
        "tyne_and_wear",
        "warwickshire",
        "west_midlands",
        "west_sussex",
        "west_yorkshire",
        "wiltshire",
        "worcestershire",
    ]

    available = regions
    available.sort()

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + england_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class GreatBritain:
    regions = ["england", "scotland", "wales"]
    england = England()

    available = regions + england.available
    available.sort()

    country = {
        "name": "great-britain" + suffix,
        "url": URL + europe_url + "great-britain" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + gb_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    for region in england.available:
        _sources[region] = {
            "name": region.replace("_", "-") + suffix,
            "url": URL + england_url + region.replace("_", "-") + suffix,
        }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Italy:
    regions = ["centro", "isole", "nord_est", "nord_ovest", "sud"]

    available = regions
    available.sort()

    country = {"name": "italy" + suffix, "url": URL + europe_url + "italy" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + italy_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Russia:
    regions = [
        "central_fed_district",
        "crimean_fed_district",
        "far_eastern_fed_district",
        "kaliningrad",
        "north_caucasus_fed_district",
        "northwestern_fed_district",
        "siberian_fed_district",
        "south_fed_district",
        "ural_fed_district",
        "volga_fed_district",
    ]

    available = regions
    available.sort()

    country = {"name": "russia" + suffix, "url": URL + russia_url + "russia" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + russia_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Poland:
    regions = [
        "dolnoslaskie",
        "kujawsko_pomorskie",
        "lodzkie",
        "lubelskie",
        "lubuskie",
        "malopolskie",
        "mazowieckie",
        "opolskie",
        "podkarpackie",
        "podlaskie",
        "pomorskie",
        "slaskie",
        "swietokrzyskie",
        "warminsko_mazurskie",
        "wielkopolskie",
        "zachodniopomorskie",
    ]

    available = regions
    available.sort()

    country = {"name": "poland" + suffix, "url": URL + europe_url + "poland" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + poland_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class BadenWuerttemberg:
    regions = [
        "freiburg_regbez",
        "karlsruhe_regbez",
        "stuttgart_regbez",
        "tuebingen_regbez",
    ]

    available = regions
    available.sort()

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + baden_wuerttemberg_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class NordrheinWestfalen:
    regions = [
        "arnsberg_regbez",
        "detmold_regbez",
        "duesseldorf_regbez",
        "koeln_regbez",
        "muenster_regbez",
    ]

    available = regions
    available.sort()

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + nordrhein_wesfalen_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Bayern:
    regions = [
        "mittelfranken",
        "niederbayern",
        "oberbayern",
        "oberfranken",
        "oberpfalz",
        "schwaben",
        "unterfranken",
    ]

    available = regions
    available.sort()

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + bayern_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Germany:
    regions = [
        "baden_wuerttemberg",
        "bayern",
        "berlin",
        "brandenburg",
        "bremen",
        "hamburg",
        "hessen",
        "mecklenburg_vorpommern",
        "niedersachsen",
        "nordrhein_westfalen",
        "rheinland_pfalz",
        "saarland",
        "sachsen_anhalt",
        "sachsen",
        "schleswig_holstein",
        "thueringen",
    ]

    baden_wuerttemberg = BadenWuerttemberg()
    bayern = Bayern()
    nordrhein_westfalen = NordrheinWestfalen()

    available = (
        regions
        + bayern.available
        + baden_wuerttemberg.available
        + nordrhein_westfalen.available
    )
    available.sort()

    country = {"name": "germany" + suffix, "url": URL + europe_url + "germany" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + germany_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    for region in nordrhein_westfalen.available:
        _sources[region] = {
            "name": region.replace("_", "-") + suffix,
            "url": URL + nordrhein_wesfalen_url + region.replace("_", "-") + suffix,
        }

    for region in baden_wuerttemberg.available:
        _sources[region] = {
            "name": region.replace("_", "-") + suffix,
            "url": URL + baden_wuerttemberg_url + region.replace("_", "-") + suffix,
        }

    for region in bayern.available:
        _sources[region] = {
            "name": region.replace("_", "-") + suffix,
            "url": URL + bayern_url + region.replace("_", "-") + suffix,
        }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Netherlands:
    regions = [
        "drenthe",
        "flevoland",
        "friesland",
        "gelderland",
        "groningen",
        "limburg",
        "noord_brabant",
        "noord_holland",
        "overijssel",
        "utrecht",
        "zeeland",
        "zuid_holland",
    ]
    available = regions
    available.sort()

    country = {
        "name": "netherlands" + suffix,
        "url": URL + europe_url + "netherlands" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + netherlands_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Canada:
    regions = [
        "alberta",
        "british_columbia",
        "manitoba",
        "new_brunswick",
        "newfoundland_and_labrador",
        "northwest_territories",
        "nova_scotia",
        "nunavut",
        "ontario",
        "prince_edward_island",
        "quebec",
        "saskatchewan",
        "yukon",
    ]
    available = regions
    available.sort()

    country = {
        "name": "canada" + suffix,
        "url": URL + north_america_url + "canada" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + canada_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Brazil:
    regions = ["centro_oeste", "nordeste", "norte", "sudeste", "sul"]
    available = regions
    available.sort()

    country = {
        "name": "brazil" + suffix,
        "url": URL + south_america_url + "brazil" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + brazil_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Japan:
    regions = [
        "chubu",
        "chugoku",
        "hokkaido",
        "kansai",
        "kanto",
        "kyushu",
        "shikoku",
        "tohoku",
    ]

    available = regions
    available.sort()

    country = {"name": "japan" + suffix, "url": URL + asia_url + "japan" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + japan_url + region + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class AustraliaOceania:
    regions = [
        "australia",
        "cook_islands",
        "fiji",
        "ile-de-clipperton",
        "kiribati",
        "marshall_islands",
        "micronesia",
        "nauru",
        "new_caledonia",
        "new_zealand",
        "niue",
        "palau",
        "papua_new_guinea",
        "pitcairn-islands",
        "polynesie-francaise",
        "samoa",
        "solomon_islands",
        "tokelau",
        "tonga",
        "tuvalu",
        "vanuatu",
        "wallis-et-futuna",
    ]

    available = regions
    available.sort()

    continent = {
        "name": "australia-oceania" + suffix,
        "url": URL + "australia-oceania" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + australia_oceania_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class NorthAmerica:
    regions = [
        "canada",
        "greenland",
        "mexico",
        "usa",
        "us_midwest",
        "us_northeast",
        "us_pacific",
        "us_south",
        "us_west",
    ]

    usa = USA()
    canada = Canada()

    available = regions
    available.sort()

    continent = {
        "name": "north-america" + suffix,
        "url": URL + "north-america" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + north_america_url + region.replace("_", "-") + suffix,
        }
        for region in regions
        if region != "usa"
    }
    # USA is "us" in GeoFabrik
    _sources["usa"] = {
        "name": "us" + suffix,
        "url": URL + north_america_url + "us" + suffix,
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class SouthAmerica:
    regions = [
        "argentina",
        "bolivia",
        "brazil",
        "chile",
        "colombia",
        "ecuador",
        "paraguay",
        "peru",
        "suriname",
        "uruguay",
        "venezuela",
    ]

    brazil = Brazil()

    available = regions
    available.sort()
    available.sort()

    continent = {
        "name": "south-america" + suffix,
        "url": URL + "south-america" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + south_america_url + region + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class CentralAmerica:
    regions = [
        "bahamas",
        "belize",
        "costa-rica",
        "cuba",
        "el-salvador",
        "guatemala",
        "haiti_and_domrep",
        "honduras",
        "jamaica",
        "nicaragua",
        "panama",
    ]

    available = regions
    available.sort()

    continent = {
        "name": "central-america" + suffix,
        "url": URL + "central-america" + suffix,
    }

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + central_america_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Europe:
    # Country specific subregions
    france = France()
    great_britain = GreatBritain()
    italy = Italy()
    russia = Russia()
    poland = Poland()
    germany = Germany()
    netherlands = Netherlands()

    regions = [
        "albania",
        "andorra",
        "austria",
        "azores",
        "belarus",
        "belgium",
        "bosnia_herzegovina",
        "bulgaria",
        "croatia",
        "cyprus",
        "czech_republic",
        "denmark",
        "estonia",
        "faroe_islands",
        "finland",
        "france",
        "georgia",
        "germany",
        "great_britain",
        "greece",
        "hungary",
        "iceland",
        "ireland_and_northern_ireland",
        "isle_of_man",
        "italy",
        "kosovo",
        "latvia",
        "liechtenstein",
        "lithuania",
        "luxembourg",
        "macedonia",
        "malta",
        "moldova",
        "monaco",
        "montenegro",
        "netherlands",
        "norway",
        "poland",
        "portugal",
        "romania",
        "russia",
        "serbia",
        "slovakia",
        "slovenia",
        "spain",
        "sweden",
        "switzerland",
        "turkey",
        "ukraine",
    ]

    available = regions
    available.sort()

    continent = {"name": "europe" + suffix, "url": URL + "europe" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + europe_url + region.replace("_", "-") + suffix,
        }
        for region in regions
        if region != "russia"
    }
    # Russia is separately from Europe
    _sources["russia"] = {"name": "russia" + suffix, "url": URL + "russia" + suffix}

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Africa:
    regions = [
        "algeria",
        "angola",
        "benin",
        "botswana",
        "burkina_faso",
        "burundi",
        "cameroon",
        "canary_islands",
        "cape_verde",
        "central_african_republic",
        "chad",
        "comores",
        "congo_brazzaville",
        "congo_democratic_republic",
        "djibouti",
        "egypt",
        "equatorial_guinea",
        "eritrea",
        "ethiopia",
        "gabon",
        "ghana",
        "guinea_bissau",
        "guinea",
        "ivory_coast",
        "kenya",
        "lesotho",
        "liberia",
        "libya",
        "madagascar",
        "malawi",
        "mali",
        "mauritania",
        "mauritius",
        "morocco",
        "mozambique",
        "namibia",
        "niger",
        "nigeria",
        "rwanda",
        "saint_helena_ascension_and_tristan_da_cunha",
        "sao_tome_and_principe",
        "senegal_and_gambia",
        "seychelles",
        "sierra_leone",
        "somalia",
        "south_africa_and_lesotho",
        "south_africa",
        "south_sudan",
        "sudan",
        "swaziland",
        "tanzania",
        "togo",
        "tunisia",
        "uganda",
        "zambia",
        "zimbabwe",
    ]

    available = regions
    available.sort()

    continent = {"name": "africa" + suffix, "url": URL + "africa" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + africa_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Asia:
    regions = [
        "afghanistan",
        "armenia",
        "azerbaijan",
        "bangladesh",
        "bhutan",
        "cambodia",
        "china",
        "east-timor",
        "gcc_states",
        "india",
        "indonesia",
        "iran",
        "iraq",
        "israel_and_palestine",
        "japan",
        "jordan",
        "kazakhstan",
        "kyrgyzstan",
        "laos",
        "lebanon",
        "malaysia_singapore_brunei",
        "maldives",
        "mongolia",
        "myanmar",
        "nepal",
        "north_korea",
        "pakistan",
        "philippines",
        "south_korea",
        "sri_lanka",
        "syria",
        "taiwan",
        "tajikistan",
        "thailand",
        "turkmenistan",
        "uzbekistan",
        "vietnam",
        "yemen",
    ]

    japan = Japan()

    available = regions
    available.sort()

    continent = {"name": "asia" + suffix, "url": URL + "asia" + suffix}

    # Create data sources
    _sources = {
        region: {
            "name": region.replace("_", "-") + suffix,
            "url": URL + asia_url + region.replace("_", "-") + suffix,
        }
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class Antarctica:
    regions = ["antarctica"]
    available = regions
    available.sort()

    continent = {"name": "antarctica" + suffix, "url": URL + "antarctica" + suffix}

    # Create data sources
    _sources = {
        region: {"name": region + suffix, "url": URL + region + suffix}
        for region in regions
    }

    __dict__ = _sources

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available


class SubRegions:
    def __init__(self):
        self.regions = [
            "brazil",
            "canada",
            "france",
            "germany",
            "great_britain",
            "italy",
            "japan",
            "netherlands",
            "poland",
            "russia",
            "usa",
        ]
        available = self.regions
        available.sort()

        self.brazil = Brazil()
        self.canada = Canada()
        self.france = France()
        self.germany = Germany()
        self.great_britain = GreatBritain()
        self.italy = Italy()
        self.japan = Japan()
        self.netherlands = Netherlands()
        self.poland = Poland()
        self.russia = Russia()
        self.usa = USA()

        self.available = {name: self.__dict__[name].available for name in self.regions}

    def __getattr__(self, name):
        return self.__dict__[name]

    def __call__(self):
        return self.available
