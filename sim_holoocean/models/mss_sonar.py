"""MSS (Mechanical Scanning Sonar) reconstruction — headless core for Layer-2.

HoloOcean 2.3.0 has NO native 360-degree mechanical scanning sonar (its
ImagingSonar / ProfilingSonar are fixed-fan, and ProfilingSonar caps at ~270 deg
azimuth). This module reconstructs an MSS by:

  - configuring a NARROW-fan ImagingSonar as a single beam (Azimuth=2 deg,
    Elevation=20 deg, AzimuthBins=1);
  - each sonar frame, calling `sensor.rotate(scan_rotation(angle))` to point the
    beam at a target azimuth;
  - accumulating single-beam returns into a polar buffer (MSSPolarMap), binned by
    angle, to assemble a full-sweep MSS image.

This realizes the paper's "scanning sonar" archetype (w_FOV up to 360 deg) for the
scanning-vs-forward dual-sonar contrast: the 360-degree scanning sonar keeps the
wall visible on every sweep, where the fixed-fan sonars cannot reach a full circle.

Implementation notes:
  - In the `env.act(...) -> sensor.rotate(...) -> env.tick()` call order, `rotate`
    takes effect within the SAME tick (no observable latency); default
    latency_ticks=0 yields exact sector-sweep alignment. (HoloOcean's documented
    "~3 tick" latency targets a different call path and does not apply here.)
  - **SCAN_AXIS = "roll" for TorpedoAUV**: the TorpedoAUV `SonarSocket` is
    "rotated +90 degrees on y-axis", so a world-yaw scan maps to the socket-local
    ROLL component. (HoveringAUV uses "yaw".) Getting this wrong sweeps the beam
    in the wrong plane.

This module is layer-isolated: it does NOT import from sim_python.
"""
from __future__ import annotations

from collections import deque

import numpy as np


# ── MSS narrow-beam ImagingSonar config (single-beam fan) ────────────────────

def mss_imaging_sonar_config(
    sensor_name: str = "MSS",
    socket: str = "SonarSocket",
    hz: float = 15.0,
    range_min: float = 0.5,
    range_max: float = 30.0,
    range_bins: int = 200,
    az_bins: int = 1,          # verified: HoloOcean 2.3.0 accepts AzimuthBins=1
    azimuth_deg: float = 2.0,  # narrow horizontal fan = single beam
    elevation_deg: float = 20.0,
    init_octree_range: float = 80.0,
    add_sigma: float = 0.05,
    mult_sigma: float = 0.05,
    range_sigma: float = 0.1,
    multipath: bool = False,
) -> dict:
    """Return the HoloOcean ImagingSonar sensor dict for the MSS single beam.

    Drop this into a scenario agent's ``sensors`` list. The beam is steered each
    frame via ``sensor.rotate(scan_rotation(angle))``; see MSSScanner.
    """
    return {
        "sensor_type": "ImagingSonar",
        "sensor_name": sensor_name,
        "socket": socket,
        "Hz": hz,
        "configuration": {
            "RangeBins": range_bins,
            "AzimuthBins": az_bins,
            "RangeMin": range_min,
            "RangeMax": range_max,
            "Azimuth": azimuth_deg,
            "Elevation": elevation_deg,
            "InitOctreeRange": init_octree_range,
            "AzimuthStreaks": -1,
            "AddSigma": add_sigma,
            "MultSigma": mult_sigma,
            "RangeSigma": range_sigma,
            "MultiPath": multipath,
            "ScrollAzimuth": 0,
            "ViewRegion": True,
            "ViewOctree": -1,
            "ShowWaring": False,
        },
    }


def scan_rotation(angle_deg: float, scan_axis: str = "roll") -> list[float]:
    """Map a desired horizontal scan angle to the sensor.rotate() [r,p,y] vector.

    TorpedoAUV: SonarSocket is rotated +90 deg on y → horizontal scan uses ROLL.
    HoveringAUV: socket is body-aligned → use "yaw". If the beam sweeps in the
    wrong plane, switch axis here.
    """
    if scan_axis == "roll":
        return [angle_deg, 0.0, 0.0]
    if scan_axis == "pitch":
        return [0.0, angle_deg, 0.0]
    return [0.0, 0.0, angle_deg]


# ── scan controller: step / direction / sector-vs-continuous / latency ───────

class MSSScanController:
    """Manages the scan angle stepping, mode switching, and latency compensation."""

    def __init__(self, start: float = -90.0, end: float = +90.0, step: float = 2.0,
                 mode: str = "sector", latency_ticks: int = 0):
        self.start = start
        self.end = end
        self.step = step
        self.mode = mode                 # 'sector' or 'continuous'
        self.latency_ticks = latency_ticks
        self.current_angle = start
        self.direction = +1
        # angle_history[0] is the target angle emitted latency_ticks ago.
        self.angle_history = deque([0.0] * (latency_ticks + 1),
                                   maxlen=latency_ticks + 1)

    def advance(self) -> float:
        if self.mode == "sector":
            self.current_angle += self.step * self.direction
            if self.current_angle >= self.end:
                self.current_angle = self.end
                self.direction = -1
            elif self.current_angle <= self.start:
                self.current_angle = self.start
                self.direction = +1
        else:  # continuous
            self.current_angle += self.step
            span = self.end - self.start
            if span <= 0:
                self.current_angle = self.start
            elif self.current_angle > self.end:
                self.current_angle = self.start + (self.current_angle - self.end) % span
        self.angle_history.append(self.current_angle)
        return self.current_angle

    def effective_angle(self) -> float:
        """Angle the current sonar frame actually corresponds to (latency-comp.)."""
        return self.angle_history[0]

    def set_latency(self, new_latency: int) -> None:
        self.latency_ticks = max(0, min(20, int(new_latency)))
        self.angle_history = deque([self.current_angle] * (self.latency_ticks + 1),
                                   maxlen=self.latency_ticks + 1)


# ── polar accumulation buffer ────────────────────────────────────────────────

class MSSPolarMap:
    """Sensor-centred polar accumulation image. Pre-allocates the full circle."""

    def __init__(self, range_bins: int, range_min: float, range_max: float,
                 angle_min: float = -180.0, angle_max: float = 180.0,
                 angle_resolution: float = 0.5):
        self.range_bins = range_bins
        self.range_min = range_min
        self.range_max = range_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.resolution = angle_resolution
        self.num_angle_bins = int(round((angle_max - angle_min) / angle_resolution)) + 1
        self.buffer = np.zeros((range_bins, self.num_angle_bins), dtype=np.float32)

    def angle_to_bin(self, angle: float) -> int | None:
        if angle < self.angle_min or angle > self.angle_max:
            return None
        return int(round((angle - self.angle_min) / self.resolution))

    def write(self, angle: float, beam_1d: np.ndarray) -> None:
        b = self.angle_to_bin(angle)
        if b is not None and 0 <= b < self.num_angle_bins:
            self.buffer[:, b] = beam_1d

    def clear(self) -> None:
        self.buffer.fill(0)


# ── convenience wrapper for headless closed-loop use (Stage 03) ──────────────

class MSSScanner:
    """Combines controller + polar map + per-frame update for headless MC use.

    Usage in a closed-loop tick (after env.tick() returns `state`):
        scanner.update(state)            # writes beam into polar map, advances scan
        sensor = env.agents[agent].sensors[scanner.sensor_name]
        sensor.rotate(scanner.next_rotation())   # steer beam for next frame

    On init, call sensor.rotate(scanner.next_rotation()) once before the loop so
    the first frame is at scan.start (not the scene-default 0 deg).
    """

    def __init__(self, sensor_name: str = "MSS", scan_axis: str = "roll",
                 start: float = -90.0, end: float = +90.0, step: float = 2.0,
                 mode: str = "sector", latency_ticks: int = 0,
                 range_min: float = 0.5, range_max: float = 30.0,
                 range_bins: int = 200, angle_resolution: float = 0.5):
        self.sensor_name = sensor_name
        self.scan_axis = scan_axis
        self.scan = MSSScanController(start, end, step, mode, latency_ticks)
        self.pmap = MSSPolarMap(range_bins, range_min, range_max,
                                angle_resolution=angle_resolution)

    def next_rotation(self) -> list[float]:
        """[r,p,y] to steer the beam to the controller's current angle."""
        return scan_rotation(self.scan.current_angle, self.scan_axis)

    def update(self, state: dict) -> float | None:
        """Process one sonar frame from `state`; returns effective angle or None.

        Writes the single-beam return into the polar map at the latency-compensated
        angle, then advances the scan for the next frame. Call sensor.rotate(
        self.next_rotation()) AFTER this to steer the beam.
        """
        if self.sensor_name not in state:
            return None
        s = np.asarray(state[self.sensor_name])  # (RangeBins, AzimuthBins)
        beam_1d = s[:, s.shape[1] // 2] if s.ndim == 2 else s
        eff = self.scan.effective_angle()
        self.pmap.write(eff, beam_1d)
        self.scan.advance()
        return eff

    @property
    def image(self) -> np.ndarray:
        """Current accumulated polar image (range_bins x angle_bins)."""
        return self.pmap.buffer
