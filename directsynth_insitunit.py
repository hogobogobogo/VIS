"""
Direct Synthesis Insitunit Generator for VocalTractLab
=======================================================
Generates vocal tract motor sequences *without* audio input.

Workflow:
  1. Build a phrase from N source→target pair movements, each shaped
     by one of several oscillator / chaotic / GENDY-style generators.
  2. Concatenate all pairs → the base phrase called the **insitunit**,
     stored as a numpy array (30 × total_frames) in memory and on disk.
  3. Generate M variations of the insitunit: each variation perturbs
     the base shape by a controlled amount, either globally or
     per-parameter (tract / glottis / individual index).
  4. Concatenate insitunit + all variations → write VTL tract-sequence
     file, optionally synthesize audio via vocaltractlab-cython.

Requirements
------------
  numpy
  vocaltractlab-cython  (optional – for audio synthesis)
  write_reaper_markers  (optional – for REAPER marker export)
"""

from __future__ import annotations

import numpy as np
import os, json, argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── optional synthesis back-end ─────────────────────────────────────────────
try:
    from vocaltractlab_cython import motor_file_to_audio_file, VtlApiError
    VTL_SYNTH_AVAILABLE = True
except ImportError:
    VTL_SYNTH_AVAILABLE = False

try:
    from write_reaper_markers import write_markers_to_wav
    MARKERS_AVAILABLE = True
except ImportError:
    MARKERS_AVAILABLE = False

# ============================================================================
# Parameter space
# ============================================================================
PARAM_NAMES = [
    "HX", "HY", "JX", "JA", "LP", "LD", "VS", "VO",
    "TCX", "TCY", "TTX", "TTY", "TBX", "TBY", "TRX", "TRY",
    "TS1", "TS2", "TS3",
    "f0", "pressure", "x_bottom", "x_top", "chink_area", "lag",
    "rel_amp", "double_pulsing", "pulse_skewness", "flutter", "aspiration_strength"
]

N_PARAMS = len(PARAM_NAMES)

lo_i = np.array([
    0.000, -6.000, -0.500, -7.000, -1.000, -2.000, 0.000, -0.100,
    -3.000, -3.000,  1.500, -3.000, -3.000, -3.000, -4.000, -6.000,
    0.000,  0.000, -1.000,
    40.000, 0.000, -0.050, -0.050, -0.250, 0.000, -1.000, 0.000,
    -0.500, 0.000, -40.000
])

hi_i = np.array([
    1.000, -3.500,  0.000, 0.000, 1.000, 4.000, 1.000, 1.000,
    4.000,  1.000,  5.500, 2.500, 4.000, 5.000, 2.000, 0.000,
    1.000,  1.000,  1.000,
    600.000, 20000.000, 0.300, 0.300, 0.250, 3.1415, 1.000, 1.000,
    0.500, 100.000, 0.000
])

centers = (lo_i + hi_i) / 2.0

TRACT_PARAMS  = list(range(0, 19))
GLOTTIS_PARAMS = list(range(19, 30))

VTL_HOP_SAMPLES = 110
AUDIO_SR        = 44100
SR_FRAMES       = AUDIO_SR / VTL_HOP_SAMPLES   # ≈ 400.9 frames / s


# ============================================================================
# Movement curve generators
# ============================================================================
# Each function takes (n_frames, **kwargs) and returns an array of shape
# (n_frames,) with values in [0, 1].  The caller maps 0→source, 1→target.

def _curve_linear(n_frames, **_):
    return np.linspace(0.0, 1.0, n_frames)


def _curve_sigmoid(n_frames, steepness=8.0, **_):
    t = np.linspace(-1.0, 1.0, n_frames)
    s = 1.0 / (1.0 + np.exp(-steepness * t))
    return (s - s[0]) / (s[-1] - s[0])


def _curve_ease_in_out(n_frames, **_):
    t = np.linspace(0.0, 1.0, n_frames)
    return t * t * (3.0 - 2.0 * t)          # smoothstep


def _curve_sine_arc(n_frames, **_):
    """Half-cosine arc: slow-start → fast-middle → slow-end"""
    t = np.linspace(0.0, 1.0, n_frames)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _curve_elastic(n_frames, oscillations=2.5, decay=4.0, **_):
    """Overshoot-and-settle elastic curve."""
    t = np.linspace(0.0, 1.0, n_frames)
    envelope = 1.0 - np.exp(-decay * t)
    ripple = np.sin(2 * np.pi * oscillations * t) * np.exp(-decay * t)
    raw = envelope + ripple * 0.3
    return np.clip(raw / raw[-1], 0.0, 1.0) if raw[-1] != 0 else raw


def _curve_logistic_chaos(n_frames, r=3.85, x0=None, chaos_depth=0.25, **_):
    """Logistic map overlay on a linear trend."""
    if x0 is None:
        x0 = np.random.uniform(0.15, 0.85)
    x = np.empty(n_frames)
    x[0] = x0
    for i in range(1, n_frames):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    trend = np.linspace(0.0, 1.0, n_frames)
    chaos = x - x.mean()
    return np.clip(trend + chaos * chaos_depth, 0.0, 1.0)


def _curve_lorenz(n_frames, sigma=10.0, rho=28.0, beta=2.667,
                  dt=0.008, chaos_depth=0.25, component=0, **_):
    """Lorenz attractor projected onto one axis."""
    xyz = np.empty((n_frames, 3))
    xyz[0] = np.random.uniform(-0.1, 0.1, 3) + [0.1, 0.1, 20.0]
    for i in range(1, n_frames):
        x, y, z = xyz[i-1]
        xyz[i, 0] = x + sigma * (y - x)        * dt
        xyz[i, 1] = y + (x * (rho - z) - y)    * dt
        xyz[i, 2] = z + (x * y - beta * z)      * dt
    vals = xyz[:, component % 3]
    mn, mx = vals.min(), vals.max()
    vals = (vals - mn) / (mx - mn) if mx > mn else vals
    trend = np.linspace(0.0, 1.0, n_frames)
    return np.clip(trend + (vals - 0.5) * chaos_depth, 0.0, 1.0)


def _curve_henon(n_frames, a=1.4, b=0.3, chaos_depth=0.25, **_):
    """Hénon map overlay."""
    x, y = np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)
    xs = []
    for _ in range(n_frames + 200):
        x_n = 1.0 - a * x * x + y
        y   = b * x
        x   = x_n
        xs.append(x)
    xs = np.array(xs[200:200 + n_frames])
    mn, mx = xs.min(), xs.max()
    xs = (xs - mn) / (mx - mn) if mx > mn else xs
    trend = np.linspace(0.0, 1.0, n_frames)
    return np.clip(trend + (xs - 0.5) * chaos_depth, 0.0, 1.0)


def _curve_duffing(n_frames, alpha=-1.0, beta=1.0, delta=0.3,
                   gamma=0.37, omega=1.2, dt=0.05,
                   chaos_depth=0.25, **_):
    """Duffing forced oscillator overlay."""
    x, v, t_v = 0.4, 0.0, 0.0
    xs = []
    for _ in range(n_frames):
        force = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t_v)
        v += force * dt
        x += v * dt
        t_v += dt
        xs.append(x)
    xs = np.array(xs)
    mn, mx = xs.min(), xs.max()
    xs = (xs - mn) / (mx - mn) if mx > mn else xs
    trend = np.linspace(0.0, 1.0, n_frames)
    return np.clip(trend + (xs - 0.5) * chaos_depth, 0.0, 1.0)


def _curve_gendy(n_frames, n_breakpoints=8,
                 amplitude_step=0.35, duration_step=0.25,
                 distribution="cauchy", seed=None, **_):
    """
    GENDY-style stochastic breakpoint trajectory (after Xenakis 1992).

    Breakpoints have (amplitude ∈ [0,1], relative duration).
    Each breakpoint is updated by a random walk drawn from `distribution`.
    The result is interpolated to n_frames and biased toward 0→1 direction.
    """
    if seed is not None:
        np.random.seed(seed)

    def _rand(dist, step):
        if dist == "cauchy":
            return np.clip(np.random.standard_cauchy() * step, -step * 3, step * 3)
        elif dist == "gaussian":
            return np.random.normal(0.0, step)
        else:  # uniform
            return np.random.uniform(-step, step)

    # Init breakpoints
    amps = np.random.uniform(0.0, 1.0, n_breakpoints)
    durs = np.abs(np.random.normal(1.0, 0.4, n_breakpoints)) + 0.1
    durs = durs / durs.sum()

    # Stochastic update (one GENDY cycle)
    amps = np.clip(amps + np.array([_rand(distribution, amplitude_step)
                                    for _ in range(n_breakpoints)]), 0.0, 1.0)
    durs = np.abs(durs + np.array([_rand(distribution, duration_step)
                                   for _ in range(n_breakpoints)]))
    durs = durs / durs.sum()

    # Build frame-level trajectory by interpolating between breakpoints
    frame_durs = np.round(durs * n_frames).astype(int)
    frame_durs[-1] += n_frames - frame_durs.sum()   # fix rounding drift
    frame_durs = np.maximum(frame_durs, 1)
    positions = np.concatenate([[0], np.cumsum(frame_durs)])
    positions = np.minimum(positions, n_frames)

    traj = np.zeros(n_frames)
    for i in range(len(amps) - 1):
        s, e = positions[i], positions[i + 1]
        if e > s:
            traj[s:e] = np.linspace(amps[i], amps[i + 1], e - s)
    if positions[-2] < n_frames:
        traj[positions[-2]:] = amps[-1]

    # Blend with linear trend to preserve src→tgt directionality
    trend  = np.linspace(0.0, 1.0, n_frames)
    result = 0.6 * traj + 0.4 * trend
    return np.clip(result, 0.0, 1.0)


def _curve_ikeda(n_frames, u=0.9, chaos_depth=0.25, **_):
    """Ikeda map (optical bistability system)."""
    x, y = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5)
    xs = []
    for _ in range(n_frames + 100):
        t_v = 0.4 - 6.0 / (1.0 + x * x + y * y)
        xn  = 1.0 + u * (x * np.cos(t_v) - y * np.sin(t_v))
        y   = u * (x * np.sin(t_v) + y * np.cos(t_v))
        x   = xn
        xs.append(x)
    xs = np.array(xs[100:100 + n_frames])
    mn, mx = xs.min(), xs.max()
    xs = (xs - mn) / (mx - mn) if mx > mn else xs
    trend = np.linspace(0.0, 1.0, n_frames)
    return np.clip(trend + (xs - 0.5) * chaos_depth, 0.0, 1.0)


def _curve_random_walk(n_frames, step_size=0.06, **_):
    """Biased random walk that arrives at 1 by the end."""
    vals = np.zeros(n_frames)
    for i in range(1, n_frames):
        progress = i / n_frames
        # Gentle pull toward the target end
        bias  = (progress - vals[i-1]) * 0.12
        delta = np.random.uniform(-step_size, step_size) + bias
        vals[i] = np.clip(vals[i-1] + delta, 0.0, 1.0)
    return vals


CURVE_GENERATORS = {
    "linear":        _curve_linear,
    "sigmoid":       _curve_sigmoid,
    "ease_in_out":   _curve_ease_in_out,
    "sine_arc":      _curve_sine_arc,
    "elastic":       _curve_elastic,
    "logistic":      _curve_logistic_chaos,
    "lorenz":        _curve_lorenz,
    "henon":         _curve_henon,
    "duffing":       _curve_duffing,
    "gendy":         _curve_gendy,
    "ikeda":         _curve_ikeda,
    "random_walk":   _curve_random_walk,
}

# Friendly display names for the GUI
GENERATOR_LABELS = {
    "linear":       "Linear",
    "sigmoid":      "Sigmoid",
    "ease_in_out":  "Ease-in/out",
    "sine_arc":     "Sine arc",
    "elastic":      "Elastic",
    "logistic":     "Logistic chaos",
    "lorenz":       "Lorenz attractor",
    "henon":        "Hénon map",
    "duffing":      "Duffing oscillator",
    "gendy":        "GENDY",
    "ikeda":        "Ikeda map",
    "random_walk":  "Random walk",
}


# ============================================================================
# Source / Target position generators
# ============================================================================

def random_position(rng=None):
    """Uniformly random position within all parameter bounds."""
    rng = rng or np.random
    pos = np.zeros(N_PARAMS)
    for i in range(N_PARAMS):
        pos[i] = rng.uniform(lo_i[i], hi_i[i])
    return pos


def center_biased_position(spread=0.3, rng=None):
    """Gaussian around parameter centres, clipped to bounds."""
    rng = rng or np.random
    pos = np.zeros(N_PARAMS)
    for i in range(N_PARAMS):
        rang = hi_i[i] - lo_i[i]
        pos[i] = np.clip(
            centers[i] + rng.normal(0.0, rang * spread),
            lo_i[i], hi_i[i]
        )
    return pos


def extreme_position(rng=None):
    """Random position biased toward the extremes of each range."""
    rng = rng or np.random
    pos = np.zeros(N_PARAMS)
    for i in range(N_PARAMS):
        if rng.random() < 0.5:
            pos[i] = lo_i[i] + rng.uniform(0, 0.2) * (hi_i[i] - lo_i[i])
        else:
            pos[i] = hi_i[i] - rng.uniform(0, 0.2) * (hi_i[i] - lo_i[i])
    return pos


POSITION_GENERATORS = {
    "random":        random_position,
    "center_biased": center_biased_position,
    "extreme":       extreme_position,
}


# ============================================================================
# Source-Target Pair
# ============================================================================

@dataclass
class SourceTargetPair:
    source:          np.ndarray          # shape (N_PARAMS,)
    target:          np.ndarray          # shape (N_PARAMS,)
    n_frames:        int
    generator_type:  str  = "linear"
    generator_params: dict = field(default_factory=dict)

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Return a trajectory array of shape (N_PARAMS, n_frames).
        Each parameter travels from source to target shaped by the generator curve.
        """
        gen_fn = CURVE_GENERATORS.get(self.generator_type, _curve_linear)
        params = dict(self.generator_params)
        if seed is not None:
            np.random.seed(seed)
            params.setdefault("seed", seed)

        traj = np.zeros((N_PARAMS, self.n_frames))
        for p in range(N_PARAMS):
            curve = gen_fn(self.n_frames, **params)
            val   = self.source[p] + (self.target[p] - self.source[p]) * curve
            traj[p, :] = np.clip(val, lo_i[p], hi_i[p])
        return traj


# ============================================================================
# Insitunit
# ============================================================================

class Insitunit:
    """
    The canonical base phrase.  Holds the (30 × T) motor array that all
    variations are derived from.
    """

    def __init__(self, pairs=None, array=None):
        self.pairs  = pairs or []
        self._array = array

    # ── generation ────────────────────────────────────────────────────────

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        if not self.pairs:
            if self._array is None:
                raise ValueError("No pairs defined and no pre-loaded array.")
            return self._array.copy()

        segs = []
        for i, pair in enumerate(self.pairs):
            pair_seed = None if seed is None else seed + i * 137
            segs.append(pair.generate(seed=pair_seed))

        self._array = np.concatenate(segs, axis=1)
        return self._array.copy()

    @property
    def array(self) -> Optional[np.ndarray]:
        return self._array

    @property
    def total_frames(self) -> int:
        return self._array.shape[1] if self._array is not None else 0

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        np.save(path, self._array)

    @classmethod
    def load(cls, path: str) -> "Insitunit":
        arr = np.load(path)
        return cls(array=arr)

    # ── variation engine ─────────────────────────────────────────────────

    def generate_variation(
        self,
        variation_depth:   float = 0.1,
        scope:             str | list = "all",
        method:            str = "additive",
        per_param_depths:  Optional[dict] = None,
        seed:              Optional[int] = None,
    ) -> np.ndarray:
        """
        Produce one variation of the insitunit.

        Parameters
        ----------
        variation_depth  : 0 = identical clone, 1 = maximum deviation.
        scope            : "all", "tract", "glottis", or list of param indices.
        method           : one of "additive", "scale_deviation",
                           "time_warp", "stochastic", "gendy_noise".
        per_param_depths : dict {param_idx: float} to override depth per param.
        seed             : reproducibility.
        """
        if self._array is None:
            raise ValueError("Insitunit array not yet generated.")

        if seed is not None:
            np.random.seed(seed)

        varied = self._array.copy()
        n_params, n_frames = varied.shape

        if scope == "all":
            param_indices = list(range(n_params))
        elif scope == "tract":
            param_indices = TRACT_PARAMS
        elif scope == "glottis":
            param_indices = GLOTTIS_PARAMS
        elif isinstance(scope, list):
            param_indices = scope
        else:
            param_indices = list(range(n_params))

        for p in param_indices:
            depth = per_param_depths.get(p, variation_depth) \
                    if per_param_depths else variation_depth
            if depth == 0.0:
                continue

            p_range = hi_i[p] - lo_i[p]

            if method == "additive":
                noise = _smooth_noise(n_frames, depth * p_range * 0.5)
                varied[p, :] = np.clip(varied[p, :] + noise, lo_i[p], hi_i[p])

            elif method == "scale_deviation":
                scale = 1.0 + np.random.uniform(-depth, depth)
                dev   = varied[p, :] - centers[p]
                varied[p, :] = np.clip(centers[p] + dev * scale, lo_i[p], hi_i[p])

            elif method == "time_warp":
                idx = _time_warp_indices(n_frames, depth)
                varied[p, :] = varied[p, idx]

            elif method == "stochastic":
                lo_s = max(0.0, 1.0 - depth)
                hi_s = 1.0 + depth
                sc   = _smooth_scale_curve(n_frames, lo_s, hi_s)
                dev  = varied[p, :] - centers[p]
                varied[p, :] = np.clip(centers[p] + dev * sc, lo_i[p], hi_i[p])

            elif method == "gendy_noise":
                n_bp  = max(4, int(8 * (1.0 - depth) + 3))
                g_traj = _curve_gendy(n_frames, n_breakpoints=n_bp,
                                      amplitude_step=depth,
                                      duration_step=depth * 0.5)
                noise = (g_traj - g_traj.mean()) * p_range * depth
                varied[p, :] = np.clip(varied[p, :] + noise, lo_i[p], hi_i[p])

        return varied


# ── helpers for the variation engine ────────────────────────────────────────

def _smooth_noise(n_frames, amplitude, control_rate_hz=3.0):
    fpc  = max(1, int(SR_FRAMES / control_rate_hz))
    nc   = int(np.ceil(n_frames / fpc)) + 1
    ctrl = np.random.uniform(-amplitude, amplitude, nc)
    xc   = np.arange(nc) * fpc
    return np.interp(np.arange(n_frames), xc, ctrl)


def _smooth_scale_curve(n_frames, lo, hi, control_rate_hz=2.0):
    fpc  = max(1, int(SR_FRAMES / control_rate_hz))
    nc   = int(np.ceil(n_frames / fpc)) + 1
    ctrl = np.random.uniform(lo, hi, nc)
    xc   = np.arange(nc) * fpc
    return np.interp(np.arange(n_frames), xc, ctrl)


def _time_warp_indices(n_frames, depth):
    """Return remapped frame indices that slightly warp the time axis."""
    t = np.linspace(0.0, 1.0, n_frames)
    fpc  = max(1, int(SR_FRAMES / 1.5))
    nc   = int(np.ceil(n_frames / fpc)) + 1
    ctrl = 1.0 + np.random.uniform(-depth, depth, nc)
    xc   = np.arange(nc) * fpc
    speed = np.interp(np.arange(n_frames), xc, ctrl)
    warped = np.cumsum(speed)
    warped = (warped - warped[0]) / (warped[-1] - warped[0]) * (n_frames - 1)
    return np.clip(np.round(warped).astype(int), 0, n_frames - 1)


# ============================================================================
# Pair factory helpers
# ============================================================================

def build_pairs(
    num_pairs:          int,
    frames_min:         int,
    frames_max:         int,
    generator_sequence: list[str],
    generator_params:   dict,
    position_mode:      str = "random",
    position_spread:    float = 0.3,
    connected:          bool = True,
    rng:                Optional[np.random.Generator] = None,
    seed:               Optional[int] = None,
) -> list[SourceTargetPair]:
    """
    Build a list of SourceTargetPair objects.

    Parameters
    ----------
    connected    : if True each source = previous target (continuous path).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    pos_fn = POSITION_GENERATORS.get(position_mode, random_position)

    # kwargs for positional generators that accept spread
    def make_pos():
        if position_mode == "center_biased":
            return pos_fn(spread=position_spread, rng=rng)
        return pos_fn(rng=rng)

    pairs = []
    prev_tgt = None

    for i in range(num_pairs):
        src = prev_tgt if (connected and prev_tgt is not None) else make_pos()
        tgt = make_pos()
        n   = int(rng.integers(frames_min, frames_max + 1))
        gen = generator_sequence[i % len(generator_sequence)]

        pair = SourceTargetPair(
            source=src.copy(),
            target=tgt.copy(),
            n_frames=n,
            generator_type=gen,
            generator_params=dict(generator_params),
        )
        pairs.append(pair)
        prev_tgt = tgt

    return pairs


# ============================================================================
# VTL file I/O
# ============================================================================

def write_vtl_file(output_path: str, motor_data: np.ndarray):
    """Write motor data (30 × T) to VTL tract-sequence text format."""
    n_frames = motor_data.shape[1]
    with open(output_path, "w") as f:
        f.write("# The first two lines (below the comment lines) indicate the name of the vocal fold model and the number of states.\n")
        f.write("# The following lines contain the control parameters of the vocal folds and the vocal tract (states)\n")
        f.write("# in steps of 110 audio samples (corresponding to about 2.5 ms for the sampling rate of 44100 Hz).\n")
        f.write("# For every step, there is one line with the vocal fold parameters followed by\n")
        f.write("# one line with the vocal tract parameters.\n")
        f.write("#\n")
        f.write("Geometric glottis\n")
        f.write(f"{n_frames}\n")
        for t in range(n_frames):
            f.write(" ".join(map(str, motor_data[19:30, t])) + "\n")
            f.write(" ".join(map(str, motor_data[0:19,  t])) + "\n")


def synthesize_audio(tractseq_path: str, audio_path: str):
    if not VTL_SYNTH_AVAILABLE:
        raise RuntimeError("vocaltractlab-cython not installed.")
    motor_file_to_audio_file(tractseq_path, audio_path)


def calculate_markers(durations: list[float]) -> list[tuple[float, str]]:
    markers, t = [], 0.0
    for i, dur in enumerate(durations):
        markers.append((t, f"Iteration {i + 1}"))
        t += dur
    return markers


# ============================================================================
# CLI entry-point
# ============================================================================

if __name__ == "__main__":
    # Force UTF-8 output on Windows (cp1252 cannot encode arrows/ticks)
    import sys as _sys, io as _io
    if hasattr(_sys.stdout, "buffer"):
        _sys.stdout = _io.TextIOWrapper(
            _sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

    parser = argparse.ArgumentParser(
        description="Direct Synthesis Insitunit Generator for VTL"
    )

    # ── phrase design ──────────────────────────────────────────────────────
    parser.add_argument("--num-pairs",       type=int,   default=6,
                        help="Number of source->target pairs in the insitunit phrase")
    parser.add_argument("--frames-min",      type=int,   default=100,
                        help="Minimum frames per source-target pair")
    parser.add_argument("--frames-max",      type=int,   default=400,
                        help="Maximum frames per source-target pair")
    parser.add_argument("--generators",      nargs="+",  default=["gendy"],
                        choices=list(CURVE_GENERATORS.keys()),
                        help="Movement generator(s); cycles if fewer than pairs")
    parser.add_argument("--position-mode",   default="random",
                        choices=list(POSITION_GENERATORS.keys()),
                        help="How source/target positions are chosen")
    parser.add_argument("--position-spread", type=float, default=0.3,
                        help="Spread for center_biased position mode (0-1)")
    parser.add_argument("--connected",       action="store_true", default=True,
                        help="Each target becomes the next source (continuous path)")
    parser.add_argument("--no-connected",    dest="connected", action="store_false")

    # ── generator-specific knobs ───────────────────────────────────────────
    parser.add_argument("--chaos-depth",     type=float, default=0.25,
                        help="Chaotic modulation depth for logistic/lorenz/henon/duffing/ikeda")
    parser.add_argument("--gendy-breakpoints", type=int, default=8,
                        help="GENDY number of breakpoints")
    parser.add_argument("--gendy-amp-step",  type=float, default=0.35,
                        help="GENDY amplitude random-walk step")
    parser.add_argument("--gendy-dur-step",  type=float, default=0.25,
                        help="GENDY duration random-walk step")
    parser.add_argument("--gendy-distribution", default="cauchy",
                        choices=["cauchy", "gaussian", "uniform"],
                        help="GENDY random distribution")

    # ── insitunit store / load ─────────────────────────────────────────────
    parser.add_argument("--load-insitunit",  type=str,   default=None,
                        help="Load a pre-saved insitunit .npy instead of generating")
    parser.add_argument("--save-insitunit",  type=str,   default=None,
                        help="Save the generated insitunit to a .npy file")

    # ── variations ────────────────────────────────────────────────────────
    parser.add_argument("--num-variations",  type=int,   default=0,
                        help="Number of variations to generate after the insitunit")
    parser.add_argument("--variation-depth", type=float, default=0.15,
                        help="Global variation depth (0=clone, 1=maximum)")
    parser.add_argument("--variation-scope", default="all",
                        choices=["all", "tract", "glottis"],
                        help="Which parameter group to vary")
    parser.add_argument("--variation-method", default="additive",
                        choices=["additive", "scale_deviation",
                                 "time_warp", "stochastic", "gendy_noise"],
                        help="Perturbation method for variations")
    parser.add_argument("--per-param-config", type=str, default=None,
                        help="JSON file mapping param_idx -> variation_depth override")

    # ── output ────────────────────────────────────────────────────────────
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--output-dir",      type=str,   default="audio/output")
    parser.add_argument("--no-synthesize",   action="store_true",
                        help="Only write tract-sequence; skip audio synthesis")
    parser.add_argument("--embed-markers",   action="store_true",
                        help="Embed iteration markers into synthesized WAV")

    args = parser.parse_args()

    rng  = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    now  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── generator params dict ──────────────────────────────────────────────
    gen_params = {
        "chaos_depth":       args.chaos_depth,
        "n_breakpoints":     args.gendy_breakpoints,
        "amplitude_step":    args.gendy_amp_step,
        "duration_step":     args.gendy_dur_step,
        "distribution":      args.gendy_distribution,
    }

    # ── load or build insitunit ────────────────────────────────────────────
    if args.load_insitunit:
        print(f"Loading insitunit from {args.load_insitunit}")
        insitunit = Insitunit.load(args.load_insitunit)
        base_array = insitunit.array
        print(f"  Loaded: {base_array.shape[1]} frames")
    else:
        print(f"\n=== Building Insitunit ===")
        print(f"  Pairs: {args.num_pairs}, generators: {args.generators}")
        print(f"  Frames per pair: {args.frames_min}-{args.frames_max}")
        print(f"  Position mode: {args.position_mode}, connected: {args.connected}")

        pairs = build_pairs(
            num_pairs=args.num_pairs,
            frames_min=args.frames_min,
            frames_max=args.frames_max,
            generator_sequence=args.generators,
            generator_params=gen_params,
            position_mode=args.position_mode,
            position_spread=args.position_spread,
            connected=args.connected,
            rng=rng,
            seed=args.seed,
        )

        for i, p in enumerate(pairs):
            print(f"  Pair {i+1}: {p.n_frames}fr  {p.generator_type}  "
                  f"src=[{', '.join(f'{v:.2f}' for v in p.source[:3])}...] "
                  f"-> tgt=[{', '.join(f'{v:.2f}' for v in p.target[:3])}...]")

        insitunit = Insitunit(pairs=pairs)
        base_array = insitunit.generate(seed=args.seed)
        print(f"  Insitunit shape: {base_array.shape}")

    # ── optionally save insitunit ──────────────────────────────────────────
    if args.save_insitunit:
        insitunit.save(args.save_insitunit)
        print(f"  Saved insitunit: {args.save_insitunit}")

    # ── per-param variation depth overrides ───────────────────────────────
    per_param_depths = None
    if args.per_param_config and os.path.exists(args.per_param_config):
        with open(args.per_param_config) as f:
            raw = json.load(f)
        per_param_depths = {int(k): float(v) for k, v in raw.items()}
        print(f"  Loaded per-param variation depths for {len(per_param_depths)} params")

    # ── assemble all sections (insitunit + variations) ────────────────────
    print(f"\n=== Assembling output ===")
    all_sections = [base_array]
    section_durations = [base_array.shape[1] * VTL_HOP_SAMPLES / AUDIO_SR]

    if args.num_variations > 0:
        print(f"  Generating {args.num_variations} variation(s)  "
              f"[depth={args.variation_depth}, scope={args.variation_scope}, "
              f"method={args.variation_method}]")

        for v in range(args.num_variations):
            v_seed = None if args.seed is None else args.seed + v + 7919
            varied = insitunit.generate_variation(
                variation_depth=args.variation_depth,
                scope=args.variation_scope,
                method=args.variation_method,
                per_param_depths=per_param_depths,
                seed=v_seed,
            )
            all_sections.append(varied)
            section_durations.append(varied.shape[1] * VTL_HOP_SAMPLES / AUDIO_SR)
            print(f"    Variation {v+1}: {varied.shape[1]} frames")

    combined = np.concatenate(all_sections, axis=1)
    total_frames = combined.shape[1]
    total_dur    = total_frames * VTL_HOP_SAMPLES / AUDIO_SR

    print(f"  Total: {total_frames} frames ({total_dur:.2f}s)")
    print(f"    = 1 insitunit + {args.num_variations} variation(s)")

    # ── write VTL tract-sequence ───────────────────────────────────────────
    suffix_parts = [f"{args.num_pairs}pairs",
                    "_".join(args.generators[:3]),
                    args.position_mode]
    if args.num_variations:
        suffix_parts.append(f"{args.num_variations}var_{args.variation_method}")
    suffix = "_".join(suffix_parts)

    tractseq_out = f"{args.output_dir}/{now}_{suffix}_tractseq.txt"
    write_vtl_file(tractseq_out, combined)
    print(f"\nSaved tract sequence: {tractseq_out}")

    # ── markers ───────────────────────────────────────────────────────────
    combined_markers = calculate_markers(section_durations)
    markers_json = tractseq_out.replace("_tractseq.txt", "_markers.json")
    markers_data = {
        "markers":    [{"time": t, "label": l} for t, l in combined_markers],
        "total_dur":  total_dur,
        "n_sections": len(all_sections),
    }
    with open(markers_json, "w") as f:
        json.dump(markers_data, f, indent=2)
    print(f"Saved markers JSON: {markers_json}")

    # ── optional audio synthesis ───────────────────────────────────────────
    if not args.no_synthesize:
        if VTL_SYNTH_AVAILABLE:
            audio_out = tractseq_out.replace("_tractseq.txt", ".wav")
            print(f"\nSynthesizing audio -> {audio_out}")
            try:
                synthesize_audio(tractseq_out, audio_out)
                print("[OK] Audio synthesized")

                if args.embed_markers and MARKERS_AVAILABLE:
                    marked_out = audio_out.replace(".wav", "_markers.wav")
                    times  = [m[0] for m in combined_markers]
                    labels = [m[1] for m in combined_markers]
                    write_markers_to_wav(audio_out, marked_out, times, labels=labels)
                    print(f"[OK] WAV with markers: {marked_out}")

            except Exception as e:
                print(f"[!] Audio synthesis failed: {e}")
        else:
            print("\nNote: vocaltractlab-cython not installed - skipping audio synthesis.")

    print("\n=== DONE ===")