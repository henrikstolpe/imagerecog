"""Thermal profile definitions for each supported vehicle class.

Each vehicle class has a simplified thermal profile defined as a set of heat zones
(engine, exhaust, tracks/wheels, body) with relative temperature offsets. Zone
positions are defined in a normalized coordinate system relative to the vehicle
bounding box:
  - x: 0.0 (left/front) to 1.0 (right/rear) along the vehicle's longitudinal axis
  - y: 0.0 (top) to 1.0 (bottom) in the vertical plane

Temperature deltas are in Kelvin, relative to ambient temperature. The renderer
will project these zones onto a 2D image plane based on viewing angle.

Requirements: 1.1, 1.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ir_recognition.models import SUPPORTED_VEHICLE_CLASSES


@dataclass(frozen=True)
class HeatZone:
    """A single heat zone within a vehicle's thermal profile.

    Attributes:
        name: Descriptive name of the zone (e.g., "engine", "exhaust").
        center_x: Normalized x-position of zone center along vehicle length [0, 1].
                   0.0 = front of vehicle, 1.0 = rear of vehicle.
        center_y: Normalized y-position of zone center in vertical plane [0, 1].
                   0.0 = top of vehicle, 1.0 = bottom of vehicle.
        center_z: Normalized z-position of zone center across vehicle width [0, 1].
                   0.0 = left side, 1.0 = right side.
        width: Normalized width of the zone (fraction of vehicle width).
        height: Normalized height of the zone (fraction of vehicle height).
        length: Normalized length of the zone (fraction of vehicle length).
        temp_delta: Temperature offset in Kelvin above ambient temperature.
        intensity: Relative emissivity/intensity factor [0, 1].
    """

    name: str
    center_x: float
    center_y: float
    center_z: float
    width: float
    height: float
    length: float
    temp_delta: float
    intensity: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Zone name must be a non-empty string")
        for attr in ("center_x", "center_y", "center_z", "width", "height", "length"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{attr} must be in [0, 1], got {val}"
                )
        if self.temp_delta < 0:
            raise ValueError(
                f"temp_delta must be >= 0, got {self.temp_delta}"
            )
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(
                f"intensity must be in [0, 1], got {self.intensity}"
            )


@dataclass(frozen=True)
class ThermalProfile:
    """Complete thermal profile for a vehicle class.

    Attributes:
        vehicle_class: Name of the vehicle class.
        zones: Sequence of heat zones defining the thermal signature.
        base_length_m: Approximate real-world length of the vehicle in meters.
        base_width_m: Approximate real-world width of the vehicle in meters.
        base_height_m: Approximate real-world height of the vehicle in meters.
    """

    vehicle_class: str
    zones: tuple[HeatZone, ...]
    base_length_m: float
    base_width_m: float
    base_height_m: float

    def __post_init__(self) -> None:
        if self.vehicle_class not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{self.vehicle_class}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        if not self.zones:
            raise ValueError("A thermal profile must have at least one heat zone")
        if self.base_length_m <= 0:
            raise ValueError(
                f"base_length_m must be > 0, got {self.base_length_m}"
            )
        if self.base_width_m <= 0:
            raise ValueError(
                f"base_width_m must be > 0, got {self.base_width_m}"
            )
        if self.base_height_m <= 0:
            raise ValueError(
                f"base_height_m must be > 0, got {self.base_height_m}"
            )

    def get_zone(self, name: str) -> HeatZone | None:
        """Retrieve a heat zone by name, or None if not found."""
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None

    def get_zones_by_type(self, zone_type: str) -> list[HeatZone]:
        """Retrieve all zones whose name contains the given type string."""
        return [z for z in self.zones if zone_type in z.name]


# ---------------------------------------------------------------------------
# T-72 Main Battle Tank
# ---------------------------------------------------------------------------
# The T-72 is a Soviet-era MBT (~9.5m long, 3.6m wide, 2.2m tall).
# Engine is rear-mounted (V-12 diesel), exhaust exits at rear.
# Tracks run along both sides. Turret is center-mounted.

T72_PROFILE = ThermalProfile(
    vehicle_class="T-72",
    zones=(
        HeatZone(
            name="engine",
            center_x=0.80,   # rear-mounted engine
            center_y=0.60,   # lower-center of hull
            center_z=0.50,   # centered
            width=0.50,
            height=0.35,
            length=0.25,
            temp_delta=45.0,  # engine runs very hot
            intensity=0.95,
        ),
        HeatZone(
            name="exhaust",
            center_x=0.95,   # exhaust exits at very rear
            center_y=0.45,
            center_z=0.35,   # offset to left side
            width=0.15,
            height=0.15,
            length=0.10,
            temp_delta=80.0,  # exhaust is hottest point
            intensity=1.0,
        ),
        HeatZone(
            name="tracks_left",
            center_x=0.50,   # runs full length
            center_y=0.85,   # bottom of vehicle
            center_z=0.10,   # left side
            width=0.15,
            height=0.20,
            length=0.85,
            temp_delta=15.0,  # friction heat from tracks
            intensity=0.7,
        ),
        HeatZone(
            name="tracks_right",
            center_x=0.50,
            center_y=0.85,
            center_z=0.90,   # right side
            width=0.15,
            height=0.20,
            length=0.85,
            temp_delta=15.0,
            intensity=0.7,
        ),
        HeatZone(
            name="turret",
            center_x=0.45,   # turret slightly forward of center
            center_y=0.25,   # top of vehicle
            center_z=0.50,
            width=0.45,
            height=0.30,
            length=0.30,
            temp_delta=8.0,   # sun-heated metal, electronics
            intensity=0.5,
        ),
        HeatZone(
            name="body",
            center_x=0.50,
            center_y=0.55,
            center_z=0.50,
            width=0.70,
            height=0.50,
            length=0.90,
            temp_delta=5.0,   # hull slightly above ambient
            intensity=0.3,
        ),
        HeatZone(
            name="gun_barrel",
            center_x=0.10,   # extends forward from turret
            center_y=0.30,
            center_z=0.50,
            width=0.05,
            height=0.05,
            length=0.30,
            temp_delta=3.0,   # minimal heat unless recently fired
            intensity=0.4,
        ),
    ),
    base_length_m=9.53,
    base_width_m=3.59,
    base_height_m=2.23,
)


# ---------------------------------------------------------------------------
# T-80 Main Battle Tank
# ---------------------------------------------------------------------------
# The T-80 is a Soviet/Russian MBT (~9.9m long, 3.6m wide, 2.2m tall).
# Gas turbine engine (rear-mounted) produces significantly more heat than
# the T-72's diesel. Distinctive hot exhaust signature.

T80_PROFILE = ThermalProfile(
    vehicle_class="T-80",
    zones=(
        HeatZone(
            name="engine",
            center_x=0.78,
            center_y=0.58,
            center_z=0.50,
            width=0.55,
            height=0.35,
            length=0.28,
            temp_delta=60.0,  # gas turbine runs hotter than diesel
            intensity=0.98,
        ),
        HeatZone(
            name="exhaust",
            center_x=0.95,
            center_y=0.40,
            center_z=0.50,   # centered exhaust
            width=0.25,
            height=0.20,
            length=0.12,
            temp_delta=110.0,  # gas turbine exhaust is extremely hot
            intensity=1.0,
        ),
        HeatZone(
            name="tracks_left",
            center_x=0.50,
            center_y=0.85,
            center_z=0.10,
            width=0.15,
            height=0.20,
            length=0.85,
            temp_delta=18.0,  # slightly hotter due to higher power
            intensity=0.7,
        ),
        HeatZone(
            name="tracks_right",
            center_x=0.50,
            center_y=0.85,
            center_z=0.90,
            width=0.15,
            height=0.20,
            length=0.85,
            temp_delta=18.0,
            intensity=0.7,
        ),
        HeatZone(
            name="turret",
            center_x=0.43,
            center_y=0.22,
            center_z=0.50,
            width=0.48,
            height=0.30,
            length=0.32,
            temp_delta=10.0,  # electronics generate more heat
            intensity=0.55,
        ),
        HeatZone(
            name="body",
            center_x=0.50,
            center_y=0.55,
            center_z=0.50,
            width=0.70,
            height=0.50,
            length=0.90,
            temp_delta=7.0,   # hull warmer due to turbine heat transfer
            intensity=0.35,
        ),
        HeatZone(
            name="gun_barrel",
            center_x=0.10,
            center_y=0.28,
            center_z=0.50,
            width=0.05,
            height=0.05,
            length=0.32,
            temp_delta=3.0,
            intensity=0.4,
        ),
    ),
    base_length_m=9.90,
    base_width_m=3.60,
    base_height_m=2.20,
)


# ---------------------------------------------------------------------------
# BMP-3 Infantry Fighting Vehicle
# ---------------------------------------------------------------------------
# The BMP-3 is a Russian IFV (~7.14m long, 3.2m wide, 2.4m tall).
# Engine is rear-mounted (UTD-29 diesel). Lower profile than MBTs.
# Smaller turret with autocannon. Lighter thermal signature overall.

BMP3_PROFILE = ThermalProfile(
    vehicle_class="BMP-3",
    zones=(
        HeatZone(
            name="engine",
            center_x=0.82,
            center_y=0.55,
            center_z=0.50,
            width=0.45,
            height=0.30,
            length=0.22,
            temp_delta=35.0,  # smaller diesel engine
            intensity=0.90,
        ),
        HeatZone(
            name="exhaust",
            center_x=0.95,
            center_y=0.50,
            center_z=0.70,   # exhaust offset to right
            width=0.12,
            height=0.12,
            length=0.08,
            temp_delta=65.0,
            intensity=0.95,
        ),
        HeatZone(
            name="tracks_left",
            center_x=0.50,
            center_y=0.88,
            center_z=0.08,
            width=0.12,
            height=0.18,
            length=0.80,
            temp_delta=12.0,  # lighter vehicle, less track friction
            intensity=0.65,
        ),
        HeatZone(
            name="tracks_right",
            center_x=0.50,
            center_y=0.88,
            center_z=0.92,
            width=0.12,
            height=0.18,
            length=0.80,
            temp_delta=12.0,
            intensity=0.65,
        ),
        HeatZone(
            name="turret",
            center_x=0.40,
            center_y=0.20,
            center_z=0.50,
            width=0.35,
            height=0.25,
            length=0.25,
            temp_delta=6.0,   # smaller turret
            intensity=0.45,
        ),
        HeatZone(
            name="body",
            center_x=0.45,
            center_y=0.55,
            center_z=0.50,
            width=0.65,
            height=0.50,
            length=0.85,
            temp_delta=4.0,
            intensity=0.25,
        ),
    ),
    base_length_m=7.14,
    base_width_m=3.23,
    base_height_m=2.40,
)


# ---------------------------------------------------------------------------
# CV90 Infantry Fighting Vehicle
# ---------------------------------------------------------------------------
# The CV90 is a Swedish IFV (~6.5m long, 3.1m wide, 2.7m tall).
# Front-mounted engine (Scania diesel). Distinctive front-heavy thermal
# signature compared to rear-engine vehicles.

CV90_PROFILE = ThermalProfile(
    vehicle_class="CV90",
    zones=(
        HeatZone(
            name="engine",
            center_x=0.20,   # front-mounted engine (distinctive!)
            center_y=0.55,
            center_z=0.50,
            width=0.50,
            height=0.35,
            length=0.25,
            temp_delta=40.0,
            intensity=0.92,
        ),
        HeatZone(
            name="exhaust",
            center_x=0.08,   # exhaust exits at front-left
            center_y=0.45,
            center_z=0.25,
            width=0.12,
            height=0.12,
            length=0.08,
            temp_delta=70.0,
            intensity=0.95,
        ),
        HeatZone(
            name="tracks_left",
            center_x=0.50,
            center_y=0.87,
            center_z=0.08,
            width=0.12,
            height=0.18,
            length=0.82,
            temp_delta=13.0,
            intensity=0.65,
        ),
        HeatZone(
            name="tracks_right",
            center_x=0.50,
            center_y=0.87,
            center_z=0.92,
            width=0.12,
            height=0.18,
            length=0.82,
            temp_delta=13.0,
            intensity=0.65,
        ),
        HeatZone(
            name="turret",
            center_x=0.55,   # turret is rear-of-center (crew compartment behind engine)
            center_y=0.20,
            center_z=0.50,
            width=0.40,
            height=0.28,
            length=0.28,
            temp_delta=7.0,
            intensity=0.50,
        ),
        HeatZone(
            name="body",
            center_x=0.50,
            center_y=0.55,
            center_z=0.50,
            width=0.65,
            height=0.50,
            length=0.88,
            temp_delta=4.0,
            intensity=0.25,
        ),
    ),
    base_length_m=6.55,
    base_width_m=3.10,
    base_height_m=2.70,
)


# ---------------------------------------------------------------------------
# Civilian Car
# ---------------------------------------------------------------------------
# A generic civilian sedan (~4.5m long, 1.8m wide, 1.5m tall).
# Front-mounted engine, exhaust runs along underside to rear.
# Much smaller and cooler thermal signature than military vehicles.
# Wheels instead of tracks, with minimal friction heat.

CIVILIAN_CAR_PROFILE = ThermalProfile(
    vehicle_class="civilian_car",
    zones=(
        HeatZone(
            name="engine",
            center_x=0.18,   # front-mounted engine
            center_y=0.50,
            center_z=0.50,
            width=0.55,
            height=0.40,
            length=0.25,
            temp_delta=30.0,  # civilian engine runs cooler
            intensity=0.85,
        ),
        HeatZone(
            name="exhaust",
            center_x=0.92,   # exhaust pipe exits at rear
            center_y=0.85,
            center_z=0.65,   # offset to one side
            width=0.08,
            height=0.08,
            length=0.06,
            temp_delta=55.0,
            intensity=0.90,
        ),
        HeatZone(
            name="exhaust_pipe",
            center_x=0.55,   # exhaust pipe runs along underside
            center_y=0.92,
            center_z=0.60,
            width=0.05,
            height=0.05,
            length=0.60,
            temp_delta=20.0,
            intensity=0.6,
        ),
        HeatZone(
            name="wheels_front",
            center_x=0.22,
            center_y=0.90,
            center_z=0.50,
            width=0.80,
            height=0.15,
            length=0.10,
            temp_delta=8.0,   # wheel/brake friction (much less than tracks)
            intensity=0.5,
        ),
        HeatZone(
            name="wheels_rear",
            center_x=0.78,
            center_y=0.90,
            center_z=0.50,
            width=0.80,
            height=0.15,
            length=0.10,
            temp_delta=6.0,   # rear wheels slightly cooler (non-drive in FWD)
            intensity=0.45,
        ),
        HeatZone(
            name="body",
            center_x=0.50,
            center_y=0.50,
            center_z=0.50,
            width=0.85,
            height=0.70,
            length=0.90,
            temp_delta=3.0,   # thin metal body, close to ambient
            intensity=0.2,
        ),
        HeatZone(
            name="windshield",
            center_x=0.35,
            center_y=0.25,
            center_z=0.50,
            width=0.70,
            height=0.25,
            length=0.15,
            temp_delta=2.0,   # glass slightly warmer from cabin heat
            intensity=0.15,
        ),
    ),
    base_length_m=4.50,
    base_width_m=1.80,
    base_height_m=1.50,
)


# ---------------------------------------------------------------------------
# Profile Registry
# ---------------------------------------------------------------------------

THERMAL_PROFILES: dict[str, ThermalProfile] = {
    "T-72": T72_PROFILE,
    "T-80": T80_PROFILE,
    "BMP-3": BMP3_PROFILE,
    "CV90": CV90_PROFILE,
    "civilian_car": CIVILIAN_CAR_PROFILE,
}


def get_thermal_profile(vehicle_class: str) -> ThermalProfile:
    """Retrieve the thermal profile for a given vehicle class.

    Args:
        vehicle_class: One of the supported vehicle classes.

    Returns:
        The ThermalProfile for the requested vehicle class.

    Raises:
        ValueError: If the vehicle class is not supported.
    """
    if vehicle_class not in THERMAL_PROFILES:
        raise ValueError(
            f"Unknown vehicle class: '{vehicle_class}'. "
            f"Supported: {list(THERMAL_PROFILES.keys())}"
        )
    return THERMAL_PROFILES[vehicle_class]


def list_vehicle_classes() -> list[str]:
    """Return a list of all supported vehicle classes with thermal profiles."""
    return list(THERMAL_PROFILES.keys())
