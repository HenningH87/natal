"""
Constants and utility functions for the natal package.
"""

from natal.config import DotDict


class Body(DotDict):
    """
    Represents a celestial body in raw data.
    Base class for all members.
    """

    name: str
    symbol: str
    value: int


class PlanetMember(Body):
    """
    Represents a planet in raw data.
    """

    ...


class AspectMember(Body):
    """
    Represents an aspect in raw data.
    (conjunction, opposition, trine, square, sextile)
    """

    ...


class ElementMember(Body):
    """
    Represents an element in raw data.
    (fire, earth, air, water)
    """

    ...


class ModalityMember(Body):
    """
    Represents a modality in raw data.
    (cardinal, fixed, mutable)
    """

    ...


class PolarityMember(Body):
    """
    Represents a polarity in raw data.
    (positive, negative)
    """

    ...


class HouseMember(Body):
    """
    Represents a house in raw data.
    """

    ...


class ExtraMember(Body):
    """
    Represents an extra celestial body in raw data.
    (e.g. asteroids, nodes)
    """

    ...


class VertexMember(Body):
    """
    Represents a vertex in raw data (asc, ic, dsc, mc).
    """

    ...


class SignMember(Body):
    """
    Represents a zodiac sign in raw data.
    """

    ruler: str
    detriment: str
    exaltation: str
    fall: str
    classic_ruler: str
    classic_detriment: str
    modality: str
    element: str
    polarity: str


# utils ==================================


def get_member(raw_data: dict, name: str) -> DotDict:
    """
    Get a member from raw data by name.

    Args:
        raw_data (dict): The raw data dictionary.
        name (str): The name of the member.

    Returns:
        DotDict: The member as a DotDict.
    """
    idx = raw_data["name"].index(name)
    member = {key: raw_data[key][idx] for key in raw_data.keys()}
    return DotDict(**member)


def get_members(raw_data: dict) -> list[DotDict]:
    """
    Get all members from raw data.

    Args:
        raw_data (dict): The raw data dictionary.

    Returns:
        list[DotDict]: A list of members as DotDicts.
    """
    return [get_member(raw_data, name) for name in raw_data["name"]]


# Raw Data ===============================

# fmt: off
PLANET_NAMES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
EXTRA_NAMES = ["asc_node", "chiron", "ceres", "pallas", "juno", "vesta"]
ELEMENT_NAMES = ["fire", "earth", "air", "water"]
MODALITY_NAMES = ["cardinal", "fixed", "mutable"]
POLARITY_NAMES = ["positive", "negative"]
SIGN_NAMES = ["aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
HOUSE_NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
ASPECT_NAMES = ["conjunction", "opposition", "trine", "square", "sextile", "quincunx"]
VERTEX_NAMES = ["asc", "ic", "dsc", "mc"]
# fmt: on

PLANETS = dict(
    name=PLANET_NAMES,
    symbol="☉☽☿♀♂♃♄♅♆♇",
    value=list(range(10)),
)

ASPECTS = dict(
    name=ASPECT_NAMES,
    symbol="☌☍△□⚹⚻",
    value=[0, 180, 120, 90, 60, 150],
)

ELEMENTS = dict(
    name=ELEMENT_NAMES,
    symbol="🜂🜃🜁🜄",
    value=[0, 1, 2, 3],
)

MODALITY = dict(
    name=MODALITY_NAMES,
    symbol="⟑⊟𛰣",
    value=[0, 1, 2],
)

POLARITY = dict(
    name=POLARITY_NAMES,
    symbol=["+", "-"],
    value=[1, -1],
)

SIGNS = dict(
    name=SIGN_NAMES,
    symbol="♈♉♊♋♌♍♎♏♐♑♒♓",
    value=list(range(1, 13)),
    ruler="mars venus mercury moon sun mercury venus pluto jupiter saturn uranus neptune".split(),
    detriment="venus pluto jupiter saturn uranus neptune mars venus mercury moon sun mercury".split(),
    exaltation=[
        "sun",
        "moon",
        "",
        "jupiter",
        "",
        "mercury",
        "saturn",
        "",
        "",
        "mars",
        "",
        "venus",
    ],
    fall=[
        "saturn",
        "",
        "",
        "mars",
        "",
        "venus",
        "sun",
        "moon",
        "",
        "jupiter",
        "",
        "mercury",
    ],
    classic_ruler="mars venus mercury moon sun mercury venus mars jupiter saturn saturn jupiter".split(),
    classic_detriment="venus mars jupiter saturn saturn jupiter mars venus mercury moon sun mercury".split(),
    modality=list(MODALITY["name"]) * 4,
    element=list(ELEMENTS["name"]) * 3,
    polarity=list(POLARITY["name"]) * 6,
)

HOUSES = dict(
    name=HOUSE_NAMES,
    symbol=[str(i) for i in range(1, 13)],
    value=list(range(1, 13)),
)

EXTRAS = dict(
    name=EXTRA_NAMES,
    symbol="☊⚷⚳⚴⚵⚶",
    value=[10, 15, 17, 18, 19, 20],
)

VERTICES = dict(
    name=VERTEX_NAMES,
    symbol=["Asc", "IC", "Dsc", "MC"],
    value=[1, 4, 7, 10],
)

# Derived Members =================================

PLANET_MEMBERS = get_members(PLANETS)
ASPECT_MEMBERS = get_members(ASPECTS)
ELEMENT_MEMBERS = get_members(ELEMENTS)
MODALITY_MEMBERS = get_members(MODALITY)
POLARITY_MEMBERS = get_members(POLARITY)
SIGN_MEMBERS = get_members(SIGNS)
HOUSE_MEMBERS = get_members(HOUSES)
EXTRA_MEMBERS = get_members(EXTRAS)
VERTEX_MEMBERS = get_members(VERTICES)
