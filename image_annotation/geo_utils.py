import math
from typing import Tuple


def haversine_distance(lat_lng_1: Tuple[float, float], lat_lng_2: Tuple[float, float], r: float = 6371000) -> float:
    """ Calculate the great-circle distance between two points on the Earth's surface."""
    lat1, lon1 = map(math.radians, lat_lng_1)
    lat2, lon2 = map(math.radians, lat_lng_2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c
