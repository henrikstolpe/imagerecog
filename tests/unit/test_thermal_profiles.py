"""Unit tests for thermal profile definitions.

Tests that all vehicle classes have valid thermal profiles with the required
heat zones (engine, exhaust, tracks/wheels, body) and realistic parameters.
"""

import pytest

from ir_recognition.ir_generator.thermal_profiles import (
    THERMAL_PROFILES,
    HeatZone,
    ThermalProfile,
    get_thermal_profile,
    list_vehicle_classes,
)
from ir_recognition.models import SUPPORTED_VEHICLE_CLASSES


class TestHeatZone:
    """Tests for the HeatZone dataclass."""

    def test_valid_zone_creation(self):
        zone = HeatZone(
            name="engine",
            center_x=0.5,
            center_y=0.5,
            center_z=0.5,
            width=0.3,
            height=0.3,
            length=0.3,
            temp_delta=40.0,
            intensity=0.9,
        )
        assert zone.name == "engine"
        assert zone.temp_delta == 40.0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            HeatZone(
                name="",
                center_x=0.5,
                center_y=0.5,
                center_z=0.5,
                width=0.3,
                height=0.3,
                length=0.3,
                temp_delta=10.0,
            )

    def test_negative_temp_delta_raises(self):
        with pytest.raises(ValueError, match="temp_delta"):
            HeatZone(
                name="engine",
                center_x=0.5,
                center_y=0.5,
                center_z=0.5,
                width=0.3,
                height=0.3,
                length=0.3,
                temp_delta=-5.0,
            )

    def test_out_of_range_position_raises(self):
        with pytest.raises(ValueError, match="center_x"):
            HeatZone(
                name="engine",
                center_x=1.5,
                center_y=0.5,
                center_z=0.5,
                width=0.3,
                height=0.3,
                length=0.3,
                temp_delta=10.0,
            )

    def test_out_of_range_intensity_raises(self):
        with pytest.raises(ValueError, match="intensity"):
            HeatZone(
                name="engine",
                center_x=0.5,
                center_y=0.5,
                center_z=0.5,
                width=0.3,
                height=0.3,
                length=0.3,
                temp_delta=10.0,
                intensity=1.5,
            )


class TestThermalProfile:
    """Tests for the ThermalProfile dataclass."""

    def test_invalid_vehicle_class_raises(self):
        zone = HeatZone(
            name="engine",
            center_x=0.5,
            center_y=0.5,
            center_z=0.5,
            width=0.3,
            height=0.3,
            length=0.3,
            temp_delta=40.0,
        )
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            ThermalProfile(
                vehicle_class="unknown",
                zones=(zone,),
                base_length_m=5.0,
                base_width_m=2.0,
                base_height_m=1.5,
            )

    def test_empty_zones_raises(self):
        with pytest.raises(ValueError, match="at least one heat zone"):
            ThermalProfile(
                vehicle_class="T-72",
                zones=(),
                base_length_m=9.5,
                base_width_m=3.6,
                base_height_m=2.2,
            )

    def test_get_zone_by_name(self):
        profile = get_thermal_profile("T-72")
        engine = profile.get_zone("engine")
        assert engine is not None
        assert engine.name == "engine"

    def test_get_zone_returns_none_for_missing(self):
        profile = get_thermal_profile("T-72")
        assert profile.get_zone("nonexistent") is None

    def test_get_zones_by_type(self):
        profile = get_thermal_profile("T-72")
        track_zones = profile.get_zones_by_type("tracks")
        assert len(track_zones) == 2


class TestProfileRegistry:
    """Tests for the profile registry and lookup functions."""

    def test_all_supported_classes_have_profiles(self):
        for vehicle_class in SUPPORTED_VEHICLE_CLASSES:
            assert vehicle_class in THERMAL_PROFILES

    def test_list_vehicle_classes(self):
        classes = list_vehicle_classes()
        assert set(classes) == set(SUPPORTED_VEHICLE_CLASSES)

    def test_get_thermal_profile_valid(self):
        profile = get_thermal_profile("T-72")
        assert profile.vehicle_class == "T-72"

    def test_get_thermal_profile_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            get_thermal_profile("M1A2")


class TestProfileCompleteness:
    """Tests that each profile has the required zone types."""

    @pytest.mark.parametrize("vehicle_class", SUPPORTED_VEHICLE_CLASSES)
    def test_has_engine_zone(self, vehicle_class):
        profile = get_thermal_profile(vehicle_class)
        engine_zones = profile.get_zones_by_type("engine")
        assert len(engine_zones) >= 1

    @pytest.mark.parametrize("vehicle_class", SUPPORTED_VEHICLE_CLASSES)
    def test_has_exhaust_zone(self, vehicle_class):
        profile = get_thermal_profile(vehicle_class)
        exhaust_zones = profile.get_zones_by_type("exhaust")
        assert len(exhaust_zones) >= 1

    @pytest.mark.parametrize("vehicle_class", SUPPORTED_VEHICLE_CLASSES)
    def test_has_tracks_or_wheels(self, vehicle_class):
        profile = get_thermal_profile(vehicle_class)
        zone_names = [z.name for z in profile.zones]
        has_tracks = any("track" in n for n in zone_names)
        has_wheels = any("wheel" in n for n in zone_names)
        assert has_tracks or has_wheels

    @pytest.mark.parametrize("vehicle_class", SUPPORTED_VEHICLE_CLASSES)
    def test_has_body_zone(self, vehicle_class):
        profile = get_thermal_profile(vehicle_class)
        body_zones = profile.get_zones_by_type("body")
        assert len(body_zones) >= 1

    @pytest.mark.parametrize("vehicle_class", SUPPORTED_VEHICLE_CLASSES)
    def test_exhaust_is_hottest(self, vehicle_class):
        """Exhaust should have the highest temperature delta in each profile."""
        profile = get_thermal_profile(vehicle_class)
        exhaust_zones = profile.get_zones_by_type("exhaust")
        max_exhaust_delta = max(z.temp_delta for z in exhaust_zones)
        # Exhaust should be hotter than body
        body_zones = profile.get_zones_by_type("body")
        max_body_delta = max(z.temp_delta for z in body_zones)
        assert max_exhaust_delta > max_body_delta

    @pytest.mark.parametrize("vehicle_class", SUPPORTED_VEHICLE_CLASSES)
    def test_positive_dimensions(self, vehicle_class):
        profile = get_thermal_profile(vehicle_class)
        assert profile.base_length_m > 0
        assert profile.base_width_m > 0
        assert profile.base_height_m > 0


class TestProfileDistinguishability:
    """Tests that profiles are distinguishable from each other."""

    def test_t80_hotter_exhaust_than_t72(self):
        """T-80 gas turbine should produce hotter exhaust than T-72 diesel."""
        t72 = get_thermal_profile("T-72")
        t80 = get_thermal_profile("T-80")
        t72_exhaust = t72.get_zone("exhaust")
        t80_exhaust = t80.get_zone("exhaust")
        assert t80_exhaust.temp_delta > t72_exhaust.temp_delta

    def test_civilian_car_smaller_than_tanks(self):
        """Civilian car should be significantly smaller than military vehicles."""
        car = get_thermal_profile("civilian_car")
        t72 = get_thermal_profile("T-72")
        assert car.base_length_m < t72.base_length_m
        assert car.base_width_m < t72.base_width_m

    def test_cv90_front_engine(self):
        """CV90 has a front-mounted engine (center_x < 0.5)."""
        cv90 = get_thermal_profile("CV90")
        engine = cv90.get_zone("engine")
        assert engine.center_x < 0.5

    def test_t72_rear_engine(self):
        """T-72 has a rear-mounted engine (center_x > 0.5)."""
        t72 = get_thermal_profile("T-72")
        engine = t72.get_zone("engine")
        assert engine.center_x > 0.5
