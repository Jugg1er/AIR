REGISTRY = {}

from .basic_controller import BasicMAC
from .ade_controller import AIRMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["air_mac"] = AIRMAC