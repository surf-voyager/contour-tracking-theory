"""Scenario builder: TorpedoAUV + forward ImagingSonar in PierHarbor.

Builds a HoloOcean `scenario_cfg` dict for the closed-loop contour-tracking demo.
Key facts (taken from the shipped PierHarbor scenarios):

  * The shipped `PierHarbor-Torpedo` and `PierHarbor-HoveringImagingSonar` scenarios
    BOTH spawn `auv0` at location [486.0, -632.0, -12.0]. The HoveringImagingSonar
    variant points an ImagingSonar (Azimuth=120, RangeMin=1, RangeMax=40) at the
    harbor wall from there — i.e. this spawn is a KNOWN sonar-visible pose near a
    real static wall (unlike the empty OpenWater world, this gives strong wall
    returns).
  * The shipped Torpedo variant uses rotation yaw=130 deg; the sonar variant uses
    yaw=180. We expose `rotation` so the driver can orient the forward sonar fan at
    the wall.

We MERGE: the validated baseline TorpedoAUV body (Fossen `torpedo` dynamics,
control_scheme=1, manualControl, CL_delta=3.0, deltaMax 15 deg, r_bg/r_bb from
models/torpedo_config.yaml) + a forward ImagingSonar on SonarSocket + PoseSensor +
DynamicsSensor (DynamicsSensor is REQUIRED by FossenInterface.update()).

This module is layer-isolated: it imports nothing from sim_python.
"""
from __future__ import annotations

from typing import Optional

# Validated sonar-visible spawn from the shipped PierHarbor scenarios.
PIERHARBOR_WALL_SPAWN = [486.0, -632.0, -12.0]
PACKAGE = "Ocean"
AGENT_NAME = "auv0"
AGENT_TYPE = "TorpedoAUV"

# Baseline dynamics + actuator (the validated REMUS-100-class torpedo baseline;
# mirrored in models/torpedo_config.yaml). Re-stated here so the scenario is
# self-contained (no YAML round-trip needed inside the driver).
BASELINE_DYNAMICS = {"r_bg": [0.0, 0.0, 0.0], "r_bb": [0.0, 0.0, -0.04]}
BASELINE_ACTUATOR = {"CL_delta": 3.0, "deltaMax_fin_deg": 15}


def forward_sonar_sensor(
    sensor_name: str = "ForwardSonar",
    hz: int = 5,
    azimuth_deg: float = 120.0,
    elevation_deg: float = 20.0,
    range_min: float = 1.0,
    range_max: float = 40.0,
    range_bins: int = 512,
    azimuth_bins: int = 256,
    init_octree_range: float = 50.0,
    add_sigma: float = 0.15,
    mult_sigma: float = 0.2,
    range_sigma: float = 0.1,
    multipath: bool = True,
) -> dict:
    """Forward ImagingSonar sensor dict.

    Defaults mirror the shipped `PierHarbor-HoveringImagingSonar` sonar (the
    proven-visible config) so the wall images reliably; azimuth/range are
    overridable from the YAML config. Hz default 5 (vs shipped 1) gives a faster
    measurement cadence for the closed loop while staying a factor of ticks_per_sec.
    """
    return {
        "sensor_type": "ImagingSonar",
        "sensor_name": sensor_name,
        "socket": "SonarSocket",
        "Hz": hz,
        "configuration": {
            "RangeBins": range_bins,
            "AzimuthBins": azimuth_bins,
            "RangeMin": range_min,
            "RangeMax": range_max,
            "InitOctreeRange": init_octree_range,
            "Elevation": elevation_deg,
            "Azimuth": azimuth_deg,
            "AzimuthStreaks": -1,
            "ScaleNoise": True,
            "AddSigma": add_sigma,
            "MultSigma": mult_sigma,
            "RangeSigma": range_sigma,
            "MultiPath": multipath,
            "ShowWarning": False,
        },
    }


def mss_sonar_sensor(
    sensor_name: str = "MSS",
    hz: float = 15.0,
    range_min: float = 1.0,
    range_max: float = 40.0,
    range_bins: int = 512,
    azimuth_deg: float = 2.0,
    elevation_deg: float = 20.0,
    init_octree_range: float = 50.0,
    add_sigma: float = 0.15,
    mult_sigma: float = 0.2,
    range_sigma: float = 0.1,
    multipath: bool = True,
) -> dict:
    """MSS single-beam ImagingSonar sensor dict (scanning arm, 360 deg).

    A NARROW-fan (Azimuth ~2 deg, AzimuthBins=1) ImagingSonar that is steered each
    frame via sensor.rotate(scan_rotation(angle)) to reconstruct a full 360 deg
    mechanical-scanning sonar (models/mss_sonar.py). Noise/range defaults mirror the
    proven-visible PierHarbor forward sonar so the wall images on the steered beam.
    """
    return {
        "sensor_type": "ImagingSonar",
        "sensor_name": sensor_name,
        "socket": "SonarSocket",
        "Hz": hz,
        "configuration": {
            "RangeBins": range_bins,
            "AzimuthBins": 1,
            "RangeMin": range_min,
            "RangeMax": range_max,
            "InitOctreeRange": init_octree_range,
            "Elevation": elevation_deg,
            "Azimuth": azimuth_deg,
            "AzimuthStreaks": -1,
            "ScaleNoise": True,
            "AddSigma": add_sigma,
            "MultSigma": mult_sigma,
            "RangeSigma": range_sigma,
            "MultiPath": multipath,
            "ShowWarning": False,
        },
    }


def build_tracking_scenario(
    world: str = "PierHarbor",
    location: Optional[list] = None,
    rotation: Optional[list] = None,
    ticks_per_sec: int = 30,
    sonar_kwargs: Optional[dict] = None,
    include_laser: bool = True,
    laser_max_distance: float = 50.0,
    include_mss: bool = False,
    mss_kwargs: Optional[dict] = None,
) -> dict:
    """Return a HoloOcean scenario_cfg dict for the closed-loop tracking demo.

    Parameters
    ----------
    world : str
        Built-in geometry-rich world ("PierHarbor" default; "Dam" also valid).
    location, rotation : list[float] or None
        Spawn pose (NWU, metres / degrees). Defaults to the validated PierHarbor
        wall-adjacent spawn with yaw oriented to put the wall in the forward fan.
    ticks_per_sec : int
        Sim tick rate; sonar Hz must divide this.
    sonar_kwargs : dict or None
        Overrides for forward_sonar_sensor().
    include_laser : bool
        Add a co-located RangeFinderSensor on SonarSocket — a reliable laser range
        channel, used as the laser fallback / ground-truth cross-check.
    include_mss : bool
        Add the MSS single-beam scanning sonar (the scanning arm). Steered each frame
        via sensor.rotate() to reconstruct a 360 deg sweep (models/mss_sonar.py).
    mss_kwargs : dict or None
        Overrides for mss_sonar_sensor().
    """
    if location is None:
        location = list(PIERHARBOR_WALL_SPAWN)
    if rotation is None:
        # yaw 150 deg: between the shipped Torpedo (130) and Sonar (180) yaws so the
        # forward sonar fan sweeps the wall the HoveringImagingSonar variant images.
        rotation = [0.0, 0.0, 150.0]
    sonar_kwargs = sonar_kwargs or {}
    mss_kwargs = mss_kwargs or {}

    sensors = [
        {"sensor_type": "PoseSensor", "socket": "IMUSocket"},
        {"sensor_type": "DynamicsSensor", "configuration": {"UseCOM": True, "UseRPY": False}},
        forward_sonar_sensor(**sonar_kwargs),
    ]
    if include_mss:
        sensors.append(mss_sonar_sensor(**mss_kwargs))
    if include_laser:
        sensors.append({
            "sensor_type": "RangeFinderSensor",
            "sensor_name": "Laser",
            "socket": "SonarSocket",
            "configuration": {
                "LaserMaxDistance": float(laser_max_distance),
                "LaserCount": 1,
                "LaserAngle": 0,
            },
        })

    agent = {
        "agent_name": AGENT_NAME,
        "agent_type": AGENT_TYPE,
        "control_scheme": 1,
        "fossen_model": "torpedo",
        "control_mode": "manualControl",
        "sensors": sensors,
        "actuator": dict(BASELINE_ACTUATOR),
        "dynamics": dict(BASELINE_DYNAMICS),
        "location": list(location),
        "rotation": list(rotation),
    }
    return {
        "name": "stage_03_wall_tracking",
        "package_name": PACKAGE,
        "world": world,
        "main_agent": AGENT_NAME,
        "ticks_per_sec": int(ticks_per_sec),
        "frames_per_sec": False,
        "octree_min": 0.02,
        "octree_max": 5.0,
        "agents": [agent],
    }


if __name__ == "__main__":
    import json

    sc = build_tracking_scenario()
    print(json.dumps(sc, indent=2))
