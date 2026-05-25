# VIS (Vocal Insistunit Synthesizer)

**Direct Synthesis Insitunit Generator for VocalTractLab**

Generate vocal tract motor sequences *without* any audio input. Instead of analysing speech, this tool builds articulatory movement trajectories from scratch using a sequence of source–target pair gestures shaped by oscillator, chaotic attractor, and GENDY-style stochastic functions. The resulting base phrase — called the **insitunit** — is stored in memory and used as the template for any number of controlled variations, all concatenated into a single VTL tract-sequence file ready for synthesis.

---

## Requirements

| Package | Installation |
|---|---|
| `numpy` | `pip install numpy` |
| `vocaltractlab-cython` | `pip install vocaltractlab-cython` |
| `write_reaper_markers` | *(optional)* for REAPER marker embedding |

> **Windows users:** Python 3.10+ recommended. The scripts force UTF-8 stdout so Windows cp1252 terminals do not cause encoding errors.

---

## Setup

Create the output folder from the script directory:

```
audio/output/
```

Place both files in the same directory:

```
directsynth_insitunit.py       ← backend / CLI
directsynth_insitunit_GUI.py   ← Tkinter GUI (run this)
```

---

## Running

### GUI (recommended)

```bash
python directsynth_insitunit_GUI.py
```

### Command line

```bash
python directsynth_insitunit.py [options]
```

---

## Core Concepts

### Source–Target Pairs

The phrase is built from **N source–target pairs**. Each pair defines:

- A **source position** — a 30-dimensional vector of vocal tract / glottis parameter values
- A **target position** — another 30-dimensional parameter vector
- A **duration** in frames (1 frame = 110 audio samples ≈ 2.5 ms at 44 100 Hz)
- A **movement generator** — the function that shapes the path from source to target

In **connected** mode (default), the target of each pair becomes the source of the next, producing a continuous path through articulatory space.

### Movement Generators

Each generator shapes how parameters travel from source to target over the pair's duration. The output is a [0, 1] curve that maps onto the source–target range.

| Generator | Category | Description |
|---|---|---|
| `linear` | Smooth | Straight-line interpolation |
| `sigmoid` | Smooth | S-curve with configurable steepness |
| `ease_in_out` | Smooth | Smoothstep cubic — slow start and end |
| `sine_arc` | Smooth | Half-cosine arc — slow extremes, fast middle |
| `elastic` | Smooth | Overshoot-and-settle spring motion |
| `logistic` | Chaotic | Logistic map overlaid on a linear trend |
| `lorenz` | Chaotic | Lorenz attractor projected onto one axis |
| `henon` | Chaotic | Hénon map overlay |
| `duffing` | Chaotic | Duffing forced oscillator |
| `ikeda` | Chaotic | Ikeda optical bistability map |
| `gendy` | Stochastic | GENDY stochastic breakpoint synthesis (Xenakis) |
| `random_walk` | Stochastic | Biased random walk that arrives at the target |

For chaotic generators, `--chaos-depth` (0–1) controls how much the attractor modulates around the smooth src→tgt trend. At `0.0` the chaos is silent; at `1.0` it dominates.

### The Insitunit

The insitunit is the **canonical base phrase** — all pairs concatenated into a `(30 × T)` motor array. It is:

- Displayed as colour-coded parameter strips in the GUI **Insitunit** tab
- Saveable as a `.npy` file and reloadable in later sessions
- Used as the template for all subsequent variations

### Variations

Variations are copies of the insitunit with controlled perturbation applied. Five perturbation methods are available:

| Method | What it does |
|---|---|
| `additive` | Adds smooth interpolated noise scaled to a fraction of each parameter's range |
| `scale_deviation` | Scales each parameter's deviation from its centre value by ±depth |
| `time_warp` | Locally compresses and expands the time axis while preserving the sequence |
| `stochastic` | A stochastic scale envelope multiplies each parameter's deviation from centre |
| `gendy_noise` | GENDY-style stochastic breakpoint noise added to the trajectory |

**Scope** controls which parameters are affected: `all`, `tract` (0–18), or `glottis` (19–29). A per-parameter JSON file can override the depth for individual parameters.

The output is: `[insitunit] + [variation 1] + [variation 2] + … + [variation N]` — all concatenated into one tract-sequence file.

---

## GUI Overview

The GUI is organised into four tabs.

### Phrase Design

- Number of source–target pairs and their frame duration range
- Position mode: `random`, `center_biased` (Gaussian around parameter centres), or `extreme` (biased toward range limits)
- Generator sequence: one row per pair; if the list is shorter than the number of pairs it cycles. Buttons for common presets (All GENDY, All Lorenz, Mix chaos)
- Generator knobs: chaos depth, GENDY breakpoints, amplitude step, duration step, GENDY distribution
- Estimated duration readout

### Insitunit

- **Generate now** — builds the insitunit in-process and displays it immediately (no subprocess needed)
- Scrollable colour-coded thumbnail of all 30 parameter trajectories (tract = green, glottis = red)
- **Save .npy** / **Load .npy** for persistence across sessions

### Variations

- Number of variations, global depth slider, scope and method selectors
- Method description panel updates to explain the selected method
- Per-parameter override table: enable/disable and set individual depth per parameter; bulk-set buttons for tract, glottis, or all

### Output & Run

- Output directory selector
- Toggle audio synthesis (requires `vocaltractlab-cython`)
- Toggle REAPER marker embedding (requires `write_reaper_markers`)
- **Use in-memory insitunit** — pins the currently displayed insitunit so the run does not regenerate from scratch
- Log output panel; the subprocess runs in a background thread so the UI stays responsive

---

## CLI Reference

### Phrase Design

```bash
--num-pairs N           # Number of source-target pairs (default: 6)
--frames-min N          # Minimum frames per pair (default: 100)
--frames-max N          # Maximum frames per pair (default: 400)
--generators G [G ...]  # Generator(s); cycles if fewer than pairs (default: gendy)
--position-mode MODE    # random | center_biased | extreme (default: random)
--position-spread F     # Spread for center_biased mode, 0-1 (default: 0.3)
--no-connected          # Do not chain target -> next source
--seed N                # Random seed for reproducibility
```

### Generator Parameters

```bash
--chaos-depth F          # Chaotic modulation depth, 0-1 (default: 0.25)
--gendy-breakpoints N    # GENDY breakpoint count (default: 8)
--gendy-amp-step F       # GENDY amplitude step size (default: 0.35)
--gendy-dur-step F       # GENDY duration step size (default: 0.25)
--gendy-distribution D   # cauchy | gaussian | uniform (default: cauchy)
```

### Insitunit Persistence

```bash
--save-insitunit PATH    # Save generated insitunit to .npy
--load-insitunit PATH    # Load a saved .npy instead of generating
```

### Variations

```bash
--num-variations N       # Number of variations (default: 0)
--variation-depth F      # Global depth, 0-1 (default: 0.15)
--variation-scope S      # all | tract | glottis (default: all)
--variation-method M     # additive | scale_deviation | time_warp |
                         # stochastic | gendy_noise (default: additive)
--per-param-config PATH  # JSON: {param_idx: depth} per-parameter overrides
```

### Output

```bash
--output-dir PATH        # Output directory (default: audio/output)
--no-synthesize          # Write tract-sequence only; skip audio synthesis
--embed-markers          # Embed REAPER markers into the synthesized WAV
```

---

## CLI Examples

```bash
# Minimal: 6 GENDY pairs, no variations, tract-sequence only
python directsynth_insitunit.py

# All Lorenz, 8 pairs, save the insitunit for reuse
python directsynth_insitunit.py \
    --num-pairs 8 \
    --generators lorenz \
    --frames-min 150 --frames-max 600 \
    --save-insitunit phrase_a.npy

# Load saved insitunit and generate 5 additive variations
python directsynth_insitunit.py \
    --load-insitunit phrase_a.npy \
    --num-variations 5 \
    --variation-depth 0.2 \
    --variation-method additive

# Mixed chaos generators, glottis-only gendy_noise variations
python directsynth_insitunit.py \
    --num-pairs 10 \
    --generators gendy lorenz henon duffing \
    --chaos-depth 0.4 \
    --num-variations 4 \
    --variation-depth 0.3 \
    --variation-scope glottis \
    --variation-method gendy_noise \
    --seed 42

# GENDY with Gaussian distribution and per-parameter override
python directsynth_insitunit.py \
    --generators gendy \
    --gendy-breakpoints 12 \
    --gendy-amp-step 0.5 \
    --gendy-distribution gaussian \
    --num-variations 3 \
    --per-param-config my_depths.json \
    --embed-markers
```

**Per-parameter config JSON format:**

```json
{
  "0":  0.30,
  "1":  0.10,
  "19": 0.05,
  "20": 0.20
}
```

Keys are parameter indices (0–29). Parameters not listed are not varied.

---

## Parameter Reference

**Tract Parameters (indices 0–18):**
```
HX  HY  JX  JA  LP  LD  VS  VO
TCX TCY TTX TTY TBX TBY TRX TRY
TS1 TS2 TS3
```

**Glottis Parameters (indices 19–29):**
```
f0  pressure  x_bottom  x_top  chink_area  lag
rel_amp  double_pulsing  pulse_skewness  flutter  aspiration_strength
```

---

## Output

All files are written to `audio/output/` (or `--output-dir`):

| File | Description |
|---|---|
| `*_tractseq.txt` | VTL tract-sequence file — load in VocalTractLab 2.3 |
| `*.wav` | Synthesized audio (requires `vocaltractlab-cython`) |
| `*_markers.json` | Insitunit / variation boundary timestamps |
| `*_markers.wav` | WAV with embedded REAPER cue markers (optional) |

The tract-sequence file is compatible with **VocalTractLab 2.3** — download at [vocaltractlab.de](https://vocaltractlab.de/index.php?page=vocaltractlab-download). VocalTractLab runs on Windows only.

---

## GENDY Notes

The GENDY generator is adapted from Iannis Xenakis's *GENDY* (GENerated DYnamics) stochastic sound synthesis algorithm (1992). In the original, a waveform is built from N breakpoints whose amplitude and duration are independently updated each cycle by random walks drawn from a probability distribution (Xenakis used Cauchy). Here the same mechanism is applied to parameter trajectories:

- **Amplitude** corresponds to the parameter's position within its source–target range
- **Duration** corresponds to the relative time width of each breakpoint segment
- The `cauchy` distribution (default) produces occasional large jumps separated by small steps — characteristic of Xenakis's original intent
- `gaussian` gives smoother, more bounded evolution
- `uniform` gives equal probability across the step range

The `--gendy-breakpoints` count controls the number of control points in the trajectory. Fewer breakpoints produce broader, slower shapes; more breakpoints produce finer texture.
