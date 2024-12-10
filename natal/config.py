from enum import StrEnum
from pydantic import BaseModel
from types import SimpleNamespace
from typing import Any, Iterator, Literal, Mapping

ThemeType = Literal["light", "dark", "mono"]


class Dictable(Mapping):
    """
    Protocols for subclasses to behave like a dict.
    """

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def update(self, other: Mapping[str, Any] | None = None, **kwargs) -> None:
        """
        Update the attributes with elements from another mapping or from key/value pairs.

        Args:
            other (Mapping[str, Any] | None): A mapping object to update from.
            **kwargs: Additional key/value pairs to update with.
        """
        if other is not None:
            for key, value in other.items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DotDict(SimpleNamespace, Dictable):
    """
    Extends SimpleNamespace to allow for unpacking and subscript notation access.
    """

    pass


class ModelDict(BaseModel, Dictable):
    """
    Extends BaseModel to allow for unpacking and subscript notation access.
    """

    # override to return keys, otherwise BaseModel.__iter__ returns key value pairs
    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)
    
class ZodiacCalculation(StrEnum):
    """Calculation for Zodiac - Sidereal used for Vedic astrology"""
    TROPICAL = "TROPICAL" 
    SIDEREAL = "SIDEREAL"


class HouseSys(StrEnum):
    Placidus = "P"
    Koch = "K"
    Equal = "E"
    Campanus = "C"
    Regiomontanus = "R"
    Porphyry = "P"
    Whole_Sign = "W"


class Orb(ModelDict):
    """default orb for natal chart"""

    conjunction: int = 10
    opposition: int = 9
    trine: int = 6
    square: int = 6
    sextile: int = 0
    quincunx: int = 0


class Theme(ModelDict):
    """
    Default colors for the chart in black and white with aspects in color.
    """
    opposition: str = "#005AB5"
    square: str = "#DC3220"
    trine: str = "#009E73"
    conjunction: str = "#F0E442"
    other_aspects: str = "#a3a2a2"
    houses: str = "#fcfcf7"
    labels: str = "#000000" 
    sign_labels: str = "#000000"
    degree_labels: str = "#000000"
    transparency: float = 0
    foreground: str = "#758492" # The colors for the lines
    background: str = "#FFFFFF"
    signWheel: str = "#ffffdb"
    horizon_color: str = "#A4BACD"
    aspectBackground: str = "#FFFFFF"


class Display(ModelDict):
    """
    Display settings for celestial bodies.
    """

    sun: bool = True
    moon: bool = True
    mercury: bool = True
    venus: bool = True
    mars: bool = True
    jupiter: bool = True
    saturn: bool = True
    uranus: bool = True
    neptune: bool = True
    pluto: bool = True
    asc_node: bool = True
    chiron: bool = True
    ceres: bool = False
    pallas: bool = False
    juno: bool = False
    vesta: bool = False
    asc: bool = True
    ic: bool = False
    dsc: bool = False
    mc: bool = True


class Chart(ModelDict):
    """
    Chart configuration settings.
    """

    stroke_width: int = 1
    stroke_opacity: float = 1
    font: str = "sans-serif"
    font_size_fraction: float = 0.55
    inner_min_degree: float = 7
    outer_min_degree: float = 6
    margin_factor: float = 0.04
    ring_thickness_fraction: float = 0.15
    spike_length_ratio: float = 0.15
    conjunction_line_multiple: float = 5
    aspect_line_ratio: float = 0.75
    horizon_line: bool = True
    # hard-coded 2.2 and 600 due to the original symbol svg size = 20x20
    scale_adj_factor: float = 900
    pos_adj_factor: float = 3.5


class Config(ModelDict):
    """
    Package configuration model.
    """

    zodiac: ZodiacCalculation = ZodiacCalculation.TROPICAL
    house_sys: HouseSys = HouseSys.Whole_Sign
    orb: Orb = Orb()
    theme: Theme = Theme()
    display: Display = Display()
    chart: Chart = Chart()
