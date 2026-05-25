"""
Direct Synthesis Insitunit GUI
================================
A Tkinter front-end for directsynth_insitunit.py.

Tabs
----
  Phrase Design   – source/target pairs, generators, position mode
  Insitunit       – view / save / load the stored base phrase
  Variations      – global and per-parameter variation settings
  Output          – synthesis, markers, run
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess, sys, os, json, threading
import numpy as np

SCRIPT_PATH = "directsynth_insitunit.py"

# ── parameter space (mirrored from backend) ─────────────────────────────────
PARAM_NAMES = [
    "HX", "HY", "JX", "JA", "LP", "LD", "VS", "VO",
    "TCX", "TCY", "TTX", "TTY", "TBX", "TBY", "TRX", "TRY",
    "TS1", "TS2", "TS3",
    "f0", "pressure", "x_bottom", "x_top", "chink_area", "lag",
    "rel_amp", "double_pulsing", "pulse_skewness", "flutter", "aspiration_strength"
]
N_PARAMS = len(PARAM_NAMES)
TRACT_PARAMS   = list(range(0, 19))
GLOTTIS_PARAMS = list(range(19, 30))

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

VTL_HOP_SAMPLES = 110
AUDIO_SR        = 44100

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
GENERATORS = list(GENERATOR_LABELS.keys())

VARIATION_METHODS = ["additive", "scale_deviation", "time_warp", "stochastic", "gendy_noise"]
POSITION_MODES    = ["random", "center_biased", "extreme"]


# ============================================================================
# Mini canvas: draws (30 × T) parameter trajectories as thumbnail strips
# ============================================================================

class InsitunitCanvas(tk.Canvas):
    """Scrollable thumbnail view of a 30-parameter motor array."""

    ROW_H   = 34
    PAD_L   = 80   # left margin for label
    PAD_R   = 10
    PAD_TOP = 8

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#1a1a2e", **kwargs)
        self._data = None          # (30, T) numpy array
        self._hover = None
        self.bind("<Motion>",   self._on_motion)
        self.bind("<Leave>",    lambda e: self._clear_hover())

    def set_data(self, array: np.ndarray):
        self._data = array
        self._render()

    def clear(self):
        self._data = None
        self.delete("all")
        w = self.winfo_width() or 600
        self.create_text(w // 2, 60, text="No insitunit in memory.",
                         fill="#555577", font=("TkDefaultFont", 11, "italic"))

    def _render(self):
        self.delete("all")
        if self._data is None:
            return
        w  = self.winfo_width()  or 700
        h  = self.winfo_height() or 600
        pw = w - self.PAD_L - self.PAD_R

        # colour strips: alternating groups
        group_colors = {
            "tract":   "#16213e",
            "glottis": "#0f3460",
        }

        n_params, n_frames = self._data.shape
        total_h = self.PAD_TOP + n_params * self.ROW_H + 4
        self.configure(scrollregion=(0, 0, w, total_h))

        for p in range(n_params):
            y_top  = self.PAD_TOP + p * self.ROW_H
            y_bot  = y_top + self.ROW_H - 2
            y_mid  = (y_top + y_bot) / 2

            # background
            bg = group_colors["glottis"] if p in GLOTTIS_PARAMS else group_colors["tract"]
            self.create_rectangle(0, y_top, w, y_bot, fill=bg, outline="", tags=f"bg_{p}")

            # parameter label
            label_color = "#5dade2" if p in TRACT_PARAMS else "#a9cce3"
            self.create_text(self.PAD_L - 4, y_mid,
                             text=f"{p:2d} {PARAM_NAMES[p]}",
                             anchor="e", fill=label_color,
                             font=("Courier", 7), tags=f"label_{p}")

            # mini waveform
            data  = self._data[p, :]
            lo, hi = lo_i[p], hi_i[p]
            rang  = hi - lo if hi != lo else 1.0
            plot_h = self.ROW_H - 6

            step  = max(1, n_frames // (pw or 1))
            pts   = []
            for i in range(0, n_frames, step):
                px = self.PAD_L + (i / n_frames) * pw
                py = y_top + 3 + (1.0 - (data[i] - lo) / rang) * plot_h
                pts.extend([px, py])

            if len(pts) >= 4:
                color = "#e74c3c" if p in GLOTTIS_PARAMS else "#2ecc71"
                self.create_line(*pts, fill=color, width=1, tags=f"wave_{p}")

            # zero-line (center of range)
            cy = y_top + 3 + 0.5 * plot_h
            self.create_line(self.PAD_L, cy, w - self.PAD_R, cy,
                             fill="#333355", width=1, dash=(2, 4), tags=f"cl_{p}")

        # column: insitunit boundary marker
        self.create_line(self.PAD_L, 0,
                         self.PAD_L, total_h,
                         fill="#444466", width=1)

    def _on_motion(self, event):
        if self._data is None:
            return
        p = (event.y - self.PAD_TOP) // self.ROW_H
        if 0 <= p < N_PARAMS and p != self._hover:
            self._hover = p
            self._render()
            # highlight hovered row
            y_top = self.PAD_TOP + p * self.ROW_H
            y_bot = y_top + self.ROW_H - 2
            w = self.winfo_width() or 700
            self.create_rectangle(0, y_top, w, y_bot,
                                  outline="#f39c12", width=1, fill="")

    def _clear_hover(self):
        if self._hover is not None:
            self._hover = None
            self._render()


# ============================================================================
# Per-parameter variation table
# ============================================================================

class PerParamVariationFrame(ttk.Frame):
    """Scrollable table: one row per parameter with depth + enable toggle."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._depths   = {}
        self._enables  = {}
        self._build()

    def _build(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Param",  width=18, font="TkDefaultFont 8 bold").pack(side="left")
        ttk.Label(hdr, text="Enable", width=7,  font="TkDefaultFont 8 bold").pack(side="left")
        ttk.Label(hdr, text="Depth",  width=8,  font="TkDefaultFont 8 bold").pack(side="left")

        canvas = tk.Canvas(self, height=260, bg="#f8f8f8")
        sb     = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        for i, name in enumerate(PARAM_NAMES):
            group_bg = "#eaf4fb" if i in TRACT_PARAMS else "#fef9e7"
            row = tk.Frame(inner, bg=group_bg)
            row.pack(fill="x", pady=1)

            tk.Label(row, text=f"{i:2d}: {name}",
                     width=18, anchor="w", bg=group_bg,
                     font=("Courier", 8)).pack(side="left")

            en = tk.BooleanVar(value=True)
            self._enables[i] = en
            tk.Checkbutton(row, variable=en, bg=group_bg).pack(side="left", padx=2)

            d = tk.DoubleVar(value=0.15)
            self._depths[i] = d
            ttk.Entry(row, textvariable=d, width=7).pack(side="left", padx=2)

        # bulk-set buttons
        btn = ttk.Frame(self)
        btn.pack(fill="x", pady=4)
        ttk.Button(btn, text="All tract 0.15",
                   command=lambda: self._bulk_set(TRACT_PARAMS, 0.15)).pack(side="left", padx=2)
        ttk.Button(btn, text="All glottis 0.10",
                   command=lambda: self._bulk_set(GLOTTIS_PARAMS, 0.10)).pack(side="left", padx=2)
        ttk.Button(btn, text="All zero",
                   command=lambda: self._bulk_set(list(range(N_PARAMS)), 0.0)).pack(side="left", padx=2)
        ttk.Button(btn, text="Enable all",
                   command=lambda: [v.set(True)  for v in self._enables.values()]).pack(side="left", padx=2)
        ttk.Button(btn, text="Disable all",
                   command=lambda: [v.set(False) for v in self._enables.values()]).pack(side="left", padx=2)

    def _bulk_set(self, indices, depth):
        for i in indices:
            self._depths[i].set(depth)

    def get_config(self) -> dict:
        return {
            i: self._depths[i].get()
            for i in range(N_PARAMS)
            if self._enables[i].get()
        }

    def load_config(self, cfg: dict):
        for k, v in cfg.items():
            i = int(k)
            if 0 <= i < N_PARAMS:
                self._depths[i].set(float(v))
                self._enables[i].set(float(v) > 0.0)


# ============================================================================
# Main GUI application
# ============================================================================

class DirectSynthGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("VTL Direct Synthesis — Insitunit Generator")
        self.geometry("1280x860")
        self.configure(bg="#1a1a2e")

        # ── state ──────────────────────────────────────────────────────────
        self._insitunit_array = None   # (30, T) stored in memory
        self._insitunit_path  = None   # last saved .npy path

        # ── phrase design vars ─────────────────────────────────────────────
        self.num_pairs       = tk.IntVar(value=6)
        self.frames_min      = tk.IntVar(value=100)
        self.frames_max      = tk.IntVar(value=400)
        self.position_mode   = tk.StringVar(value="random")
        self.position_spread = tk.DoubleVar(value=0.30)
        self.connected       = tk.BooleanVar(value=True)
        self.seed            = tk.StringVar(value="")

        # generator sequence (list of comboboxes)
        self._gen_rows   = []   # list of (frame, StringVar)
        self._gen_count  = tk.IntVar(value=1)

        # generator knobs
        self.chaos_depth        = tk.DoubleVar(value=0.25)
        self.gendy_breakpoints  = tk.IntVar(value=8)
        self.gendy_amp_step     = tk.DoubleVar(value=0.35)
        self.gendy_dur_step     = tk.DoubleVar(value=0.25)
        self.gendy_distribution = tk.StringVar(value="cauchy")

        # ── variation vars ──────────────────────────────────────────────────
        self.num_variations    = tk.IntVar(value=3)
        self.variation_depth   = tk.DoubleVar(value=0.15)
        self.variation_scope   = tk.StringVar(value="all")
        self.variation_method  = tk.StringVar(value="additive")
        self.use_per_param_var = tk.BooleanVar(value=False)

        # ── output vars ────────────────────────────────────────────────────
        self.output_dir       = tk.StringVar(value="audio/output")
        self.no_synthesize    = tk.BooleanVar(value=False)
        self.embed_markers    = tk.BooleanVar(value=False)

        self._build_ui()

    # ── UI build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("TNotebook", background="#1a1a2e")
        style.configure("TNotebook.Tab", padding=[10, 4])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        tab_phrase   = ttk.Frame(nb)
        tab_insitunit = ttk.Frame(nb)
        tab_var      = ttk.Frame(nb)
        tab_output   = ttk.Frame(nb)

        nb.add(tab_phrase,    text="  Phrase Design  ")
        nb.add(tab_insitunit, text="  Insitunit  ")
        nb.add(tab_var,       text="  Variations  ")
        nb.add(tab_output,    text="  Output & Run  ")

        self._build_phrase_tab(tab_phrase)
        self._build_insitunit_tab(tab_insitunit)
        self._build_variations_tab(tab_var)
        self._build_output_tab(tab_output)

    # ── Tab 1: Phrase Design ─────────────────────────────────────────────────

    def _build_phrase_tab(self, parent):
        pad = dict(padx=8, pady=4)
        left  = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, **pad)
        right = ttk.Frame(parent)
        right.pack(side="right", fill="both", expand=True, **pad)

        # ── Pair structure ───────────────────────────────────────────────
        struc = ttk.LabelFrame(left, text="Pair Structure")
        struc.pack(fill="x", **pad)

        ttk.Label(struc, text="Number of pairs:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(struc, from_=1, to=64, textvariable=self.num_pairs,
                    width=6, command=self._refresh_generator_rows).grid(row=0, column=1, sticky="w")

        ttk.Label(struc, text="Frames per pair (min):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(struc, textvariable=self.frames_min, width=7).grid(row=1, column=1, sticky="w")

        ttk.Label(struc, text="Frames per pair (max):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(struc, textvariable=self.frames_max, width=7).grid(row=2, column=1, sticky="w")

        ttk.Label(struc, text="Position mode:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(struc, textvariable=self.position_mode, values=POSITION_MODES,
                     state="readonly", width=14).grid(row=3, column=1, sticky="w")

        ttk.Label(struc, text="Center spread (0–1):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Scale(struc, variable=self.position_spread, from_=0.0, to=1.0,
                  orient="horizontal", length=120).grid(row=4, column=1, sticky="w")

        ttk.Checkbutton(struc, text="Connected path (target → next source)",
                        variable=self.connected).grid(row=5, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(struc, text="Seed (blank = random):").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(struc, textvariable=self.seed, width=10).grid(row=6, column=1, sticky="w")

        # ── Generator sequence ───────────────────────────────────────────
        gen_outer = ttk.LabelFrame(left, text="Generator Sequence  (one row per pair; cycles if shorter)")
        gen_outer.pack(fill="both", expand=True, **pad)

        gen_toolbar = ttk.Frame(gen_outer)
        gen_toolbar.pack(fill="x")
        ttk.Button(gen_toolbar, text="+ Row",   command=self._add_gen_row).pack(side="left", padx=2)
        ttk.Button(gen_toolbar, text="− Row",   command=self._remove_gen_row).pack(side="left", padx=2)
        ttk.Button(gen_toolbar, text="All GENDY",
                   command=lambda: self._fill_all_gen("gendy")).pack(side="left", padx=4)
        ttk.Button(gen_toolbar, text="All Lorenz",
                   command=lambda: self._fill_all_gen("lorenz")).pack(side="left", padx=2)
        ttk.Button(gen_toolbar, text="Mix chaos",
                   command=self._fill_mix_chaos).pack(side="left", padx=2)

        gen_canvas = tk.Canvas(gen_outer, height=220, bg="#f0f0f0")
        gen_sb     = ttk.Scrollbar(gen_outer, orient="vertical", command=gen_canvas.yview)
        self._gen_inner = ttk.Frame(gen_canvas)
        self._gen_inner.bind("<Configure>",
                             lambda e: gen_canvas.configure(
                                 scrollregion=gen_canvas.bbox("all")))
        gen_canvas.create_window((0, 0), window=self._gen_inner, anchor="nw")
        gen_canvas.configure(yscrollcommand=gen_sb.set)
        gen_canvas.pack(side="left", fill="both", expand=True)
        gen_sb.pack(side="right", fill="y")

        # seed initial row
        self._add_gen_row(init_value="gendy")

        # ── Generator knobs ──────────────────────────────────────────────
        knobs = ttk.LabelFrame(right, text="Generator Parameters")
        knobs.pack(fill="x", **pad)

        def _row(lbl, var, r, width=8):
            ttk.Label(knobs, text=lbl).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(knobs, textvariable=var, width=width).grid(row=r, column=1, sticky="w")

        _row("Chaos depth (0–1):",          self.chaos_depth,       0)
        _row("GENDY breakpoints:",          self.gendy_breakpoints, 1)
        _row("GENDY amplitude step:",       self.gendy_amp_step,    2)
        _row("GENDY duration step:",        self.gendy_dur_step,    3)
        ttk.Label(knobs, text="GENDY distribution:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Combobox(knobs, textvariable=self.gendy_distribution,
                     values=["cauchy", "gaussian", "uniform"],
                     state="readonly", width=10).grid(row=4, column=1, sticky="w")

        # ── Quick preview estimate ───────────────────────────────────────
        preview_frame = ttk.LabelFrame(right, text="Estimated Duration")
        preview_frame.pack(fill="x", **pad)

        self._dur_label = ttk.Label(preview_frame,
                                    text="—",
                                    font=("TkDefaultFont", 10, "bold"))
        self._dur_label.pack(pady=6)
        ttk.Button(preview_frame, text="Estimate", command=self._estimate_duration).pack()

    def _add_gen_row(self, init_value=None):
        idx = len(self._gen_rows)
        frm = ttk.Frame(self._gen_inner)
        frm.pack(fill="x", pady=1)

        ttk.Label(frm, text=f"Pair {idx + 1}:", width=8).pack(side="left")
        var = tk.StringVar(value=init_value or "gendy")
        cb  = ttk.Combobox(frm, textvariable=var,
                            values=[f"{k}  ({v})" for k, v in GENERATOR_LABELS.items()],
                            state="readonly", width=22)
        # Combobox shows "key  (label)" – extract key on use
        cb.pack(side="left", padx=2)
        self._gen_rows.append((frm, var))

    def _remove_gen_row(self):
        if len(self._gen_rows) > 1:
            frm, _ = self._gen_rows.pop()
            frm.destroy()

    def _refresh_generator_rows(self):
        n = self.num_pairs.get()
        while len(self._gen_rows) < n:
            self._add_gen_row(init_value=self._gen_rows[-1][1].get().split()[0]
                              if self._gen_rows else "gendy")
        while len(self._gen_rows) > max(1, n):
            self._remove_gen_row()

    def _fill_all_gen(self, gen_key):
        for _, var in self._gen_rows:
            var.set(f"{gen_key}  ({GENERATOR_LABELS[gen_key]})")

    def _fill_mix_chaos(self):
        chaos = ["logistic", "lorenz", "henon", "duffing", "ikeda", "gendy"]
        for i, (_, var) in enumerate(self._gen_rows):
            k = chaos[i % len(chaos)]
            var.set(f"{k}  ({GENERATOR_LABELS[k]})")

    def _get_generator_sequence(self) -> list[str]:
        result = []
        for _, var in self._gen_rows:
            raw = var.get().split()[0]
            if raw in GENERATOR_LABELS:
                result.append(raw)
        return result or ["gendy"]

    def _estimate_duration(self):
        avg = (self.frames_min.get() + self.frames_max.get()) / 2
        total_frames = avg * self.num_pairs.get()
        dur = total_frames * VTL_HOP_SAMPLES / AUDIO_SR
        n_var = 0
        try:
            n_var = int(self.num_variations.get())
        except Exception:
            pass
        total = dur * (1 + n_var)
        self._dur_label.config(
            text=f"Insitunit ≈ {dur:.1f}s  |  +{n_var} var ≈ {total:.1f}s total"
        )

    # ── Tab 2: Insitunit ─────────────────────────────────────────────────────

    def _build_insitunit_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=6)

        self._insitunit_status = ttk.Label(
            top, text="No insitunit in memory.",
            font=("TkDefaultFont", 10, "bold"), foreground="#c0392b"
        )
        self._insitunit_status.pack(side="left")

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", padx=8, pady=2)
        ttk.Button(btn_row, text="Generate now",
                   command=self._generate_insitunit_preview).pack(side="left", padx=3)
        ttk.Button(btn_row, text="Save .npy",
                   command=self._save_insitunit).pack(side="left", padx=3)
        ttk.Button(btn_row, text="Load .npy",
                   command=self._load_insitunit).pack(side="left", padx=3)
        ttk.Button(btn_row, text="Clear",
                   command=self._clear_insitunit).pack(side="left", padx=3)

        # Canvas with v-scrollbar
        cv_frame = ttk.Frame(parent)
        cv_frame.pack(fill="both", expand=True, padx=8, pady=4)

        vsb = ttk.Scrollbar(cv_frame, orient="vertical")
        self._insitunit_canvas = InsitunitCanvas(cv_frame,
                                                  width=800, height=500,
                                                  yscrollcommand=vsb.set)
        vsb.config(command=self._insitunit_canvas.yview)
        self._insitunit_canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._insitunit_canvas.clear()

    def _generate_insitunit_preview(self):
        """
        Quick in-process preview: build the insitunit without calling the
        subprocess, so we can display it immediately.
        """
        try:
            from directsynth_insitunit import (
                build_pairs, Insitunit, CURVE_GENERATORS,
                POSITION_GENERATORS, random_position, center_biased_position, extreme_position
            )
        except ImportError:
            # Fall back to inline generation if the backend is not importable yet
            self._generate_insitunit_via_subprocess()
            return

        seed_str = self.seed.get().strip()
        seed     = int(seed_str) if seed_str.isdigit() else None
        rng      = np.random.default_rng(seed)

        gen_params = {
            "chaos_depth":    self.chaos_depth.get(),
            "n_breakpoints":  self.gendy_breakpoints.get(),
            "amplitude_step": self.gendy_amp_step.get(),
            "duration_step":  self.gendy_dur_step.get(),
            "distribution":   self.gendy_distribution.get(),
        }

        pairs = build_pairs(
            num_pairs=self.num_pairs.get(),
            frames_min=self.frames_min.get(),
            frames_max=self.frames_max.get(),
            generator_sequence=self._get_generator_sequence(),
            generator_params=gen_params,
            position_mode=self.position_mode.get(),
            position_spread=self.position_spread.get(),
            connected=self.connected.get(),
            rng=rng,
            seed=seed,
        )

        insitunit = Insitunit(pairs=pairs)
        arr = insitunit.generate(seed=seed)

        self._insitunit_array = arr
        dur = arr.shape[1] * VTL_HOP_SAMPLES / AUDIO_SR
        self._insitunit_status.config(
            text=f"Insitunit in memory — {arr.shape[1]} frames ({dur:.2f}s)",
            foreground="#27ae60"
        )
        self._insitunit_canvas.set_data(arr)

    def _generate_insitunit_via_subprocess(self):
        """Fallback: generate via subprocess and load back the .npy."""
        import tempfile
        tmp_npy = tempfile.mktemp(suffix=".npy")
        cmd = self._build_cmd(save_insitunit=tmp_npy, no_variations=True)
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(tmp_npy):
                arr = np.load(tmp_npy)
                self._insitunit_array = arr
                dur = arr.shape[1] * VTL_HOP_SAMPLES / AUDIO_SR
                self._insitunit_status.config(
                    text=f"Insitunit in memory — {arr.shape[1]} frames ({dur:.2f}s)",
                    foreground="#27ae60"
                )
                self._insitunit_canvas.set_data(arr)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_insitunit(self):
        if self._insitunit_array is None:
            messagebox.showwarning("No data", "Generate an insitunit first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("Numpy array", "*.npy")],
            title="Save Insitunit"
        )
        if path:
            np.save(path, self._insitunit_array)
            self._insitunit_path = path
            messagebox.showinfo("Saved", f"Insitunit saved to:\n{path}")

    def _load_insitunit(self):
        path = filedialog.askopenfilename(
            filetypes=[("Numpy array", "*.npy")],
            title="Load Insitunit"
        )
        if not path:
            return
        try:
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[0] != 30:
                raise ValueError(f"Expected shape (30, T), got {arr.shape}")
            self._insitunit_array = arr
            self._insitunit_path  = path
            dur = arr.shape[1] * VTL_HOP_SAMPLES / AUDIO_SR
            self._insitunit_status.config(
                text=f"Loaded: {path}  — {arr.shape[1]} frames ({dur:.2f}s)",
                foreground="#2980b9"
            )
            self._insitunit_canvas.set_data(arr)
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _clear_insitunit(self):
        self._insitunit_array = None
        self._insitunit_path  = None
        self._insitunit_status.config(
            text="No insitunit in memory.", foreground="#c0392b"
        )
        self._insitunit_canvas.clear()

    # ── Tab 3: Variations ───────────────────────────────────────────────────

    def _build_variations_tab(self, parent):
        pad = dict(padx=8, pady=4)
        left  = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, **pad)
        right = ttk.Frame(parent)
        right.pack(side="right", fill="both", expand=True, **pad)

        # ── Global settings ──────────────────────────────────────────────
        glob = ttk.LabelFrame(left, text="Global Variation Settings")
        glob.pack(fill="x", **pad)

        def grow(lbl, wid, r):
            ttk.Label(glob, text=lbl).grid(row=r, column=0, sticky="w", **pad)
            return wid

        ttk.Label(glob, text="Number of variations:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(glob, from_=0, to=100, textvariable=self.num_variations,
                    width=6).grid(row=0, column=1, sticky="w")

        ttk.Label(glob, text="Variation depth (0–1):").grid(row=1, column=0, sticky="w", **pad)
        depth_row = ttk.Frame(glob)
        depth_row.grid(row=1, column=1, sticky="w")
        ttk.Scale(depth_row, variable=self.variation_depth,
                  from_=0.0, to=1.0, orient="horizontal",
                  length=140).pack(side="left")
        ttk.Label(depth_row, textvariable=self.variation_depth,
                  width=5).pack(side="left")

        ttk.Label(glob, text="Scope:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(glob, textvariable=self.variation_scope,
                     values=["all", "tract", "glottis"],
                     state="readonly", width=12).grid(row=2, column=1, sticky="w")

        ttk.Label(glob, text="Method:").grid(row=3, column=0, sticky="w", **pad)
        method_cb = ttk.Combobox(glob, textvariable=self.variation_method,
                                  values=VARIATION_METHODS,
                                  state="readonly", width=16)
        method_cb.grid(row=3, column=1, sticky="w")

        # ── Method descriptions ──────────────────────────────────────────
        desc_frame = ttk.LabelFrame(left, text="Method Description")
        desc_frame.pack(fill="x", **pad)
        self._method_desc = ttk.Label(desc_frame, text="", wraplength=340,
                                       font=("TkDefaultFont", 8),
                                       foreground="#555")
        self._method_desc.pack(padx=6, pady=4)

        DESCS = {
            "additive":        "Smooth noise added to the insitunit trajectory. Depth scales noise amplitude as a fraction of each parameter's range.",
            "scale_deviation": "Each parameter's deviation from its centre is scaled by (1 ± depth). Preserves the shape but alters its magnitude.",
            "time_warp":       "The time axis is locally compressed/expanded. Depth controls warp intensity. Pitch and rhythm distort but the sequence is preserved.",
            "stochastic":      "A stochastic scale envelope multiplies each parameter's deviation. Similar to the original stochastic-scaling mode.",
            "gendy_noise":     "GENDY-style stochastic breakpoint noise is added. The number of breakpoints inversely tracks depth for organic variation.",
        }

        def _update_desc(*_):
            self._method_desc.config(text=DESCS.get(self.variation_method.get(), ""))

        self.variation_method.trace("w", _update_desc)
        _update_desc()

        # ── Per-parameter override ───────────────────────────────────────
        pp_toggle = ttk.LabelFrame(left, text="Per-Parameter Override")
        pp_toggle.pack(fill="x", **pad)
        ttk.Checkbutton(pp_toggle,
                         text="Enable per-parameter variation depth",
                         variable=self.use_per_param_var).pack(anchor="w", padx=6)

        # ── Per-parameter table (right column) ───────────────────────────
        pp_outer = ttk.LabelFrame(right, text="Per-Parameter Variation Depths")
        pp_outer.pack(fill="both", expand=True, **pad)

        self._per_param_var_frame = PerParamVariationFrame(pp_outer)
        self._per_param_var_frame.pack(fill="both", expand=True)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", **pad)
        ttk.Button(btn_row, text="Save config",
                   command=self._save_var_config).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Load config",
                   command=self._load_var_config).pack(side="left", padx=2)

    def _save_var_config(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if path:
            with open(path, "w") as f:
                json.dump(self._per_param_var_frame.get_config(), f, indent=2)
            messagebox.showinfo("Saved", f"Saved to {path}")

    def _load_var_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                with open(path) as f:
                    cfg = json.load(f)
                self._per_param_var_frame.load_config(cfg)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ── Tab 4: Output & Run ─────────────────────────────────────────────────

    def _build_output_tab(self, parent):
        pad = dict(padx=8, pady=4)
        left  = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, **pad)
        right = ttk.Frame(parent)
        right.pack(side="right", fill="both", expand=True, **pad)

        # Output dir
        od = ttk.LabelFrame(left, text="Output Directory")
        od.pack(fill="x", **pad)
        od_row = ttk.Frame(od)
        od_row.pack(fill="x", padx=6, pady=4)
        ttk.Entry(od_row, textvariable=self.output_dir, width=36).pack(side="left")
        ttk.Button(od_row, text="Browse",
                   command=lambda: self.output_dir.set(
                       filedialog.askdirectory() or self.output_dir.get()
                   )).pack(side="left", padx=4)

        # Synthesis options
        syn = ttk.LabelFrame(left, text="Audio Synthesis")
        syn.pack(fill="x", **pad)
        ttk.Checkbutton(syn, text="Skip audio synthesis (tract-sequence only)",
                         variable=self.no_synthesize).pack(anchor="w", padx=6)
        ttk.Checkbutton(syn, text="Embed REAPER markers into WAV",
                         variable=self.embed_markers).pack(anchor="w", padx=6)

        # Insitunit source for the run
        ins = ttk.LabelFrame(left, text="Insitunit Source for this Run")
        ins.pack(fill="x", **pad)
        self._run_insitunit_label = ttk.Label(
            ins, text="Will generate fresh from Phrase Design settings.",
            foreground="#555", font=("TkDefaultFont", 8, "italic")
        )
        self._run_insitunit_label.pack(padx=6, pady=4, anchor="w")
        ttk.Button(ins, text="Use in-memory insitunit",
                   command=self._set_run_from_memory).pack(side="left", padx=6, pady=4)
        ttk.Button(ins, text="Clear (regenerate)",
                   command=self._clear_run_insitunit).pack(side="left")

        self._run_use_memory_npy = None  # set to npy path when memory insitunit is used

        # Run button
        run_btn = tk.Button(
            left, text="▶  GENERATE & SYNTHESIZE",
            command=self._run,
            bg="#27ae60", fg="white",
            font=("TkDefaultFont", 13, "bold"),
            relief="flat", padx=12, pady=8
        )
        run_btn.pack(pady=14)

        # Log
        log_frm = ttk.LabelFrame(right, text="Log")
        log_frm.pack(fill="both", expand=True)
        self._log = scrolledtext.ScrolledText(log_frm, height=36, width=58,
                                               font=("Courier", 8))
        self._log.pack(fill="both", expand=True, padx=4, pady=4)

    def _set_run_from_memory(self):
        if self._insitunit_array is None:
            messagebox.showwarning("No insitunit", "Generate or load an insitunit first.")
            return
        import tempfile
        tmp = tempfile.mktemp(suffix=".npy")
        np.save(tmp, self._insitunit_array)
        self._run_use_memory_npy = tmp
        self._run_insitunit_label.config(
            text=f"Will use in-memory insitunit ({self._insitunit_array.shape[1]} frames).",
            foreground="#27ae60"
        )

    def _clear_run_insitunit(self):
        self._run_use_memory_npy = None
        self._run_insitunit_label.config(
            text="Will generate fresh from Phrase Design settings.",
            foreground="#555"
        )

    # ── Command builder ─────────────────────────────────────────────────────

    def _build_cmd(self, save_insitunit=None, no_variations=False,
                   load_insitunit=None) -> list[str]:
        cmd = [sys.executable, SCRIPT_PATH]

        if load_insitunit or self._run_use_memory_npy:
            npy = load_insitunit or self._run_use_memory_npy
            cmd += ["--load-insitunit", npy]
        else:
            cmd += [
                "--num-pairs",       str(self.num_pairs.get()),
                "--frames-min",      str(self.frames_min.get()),
                "--frames-max",      str(self.frames_max.get()),
                "--position-mode",   self.position_mode.get(),
                "--position-spread", str(self.position_spread.get()),
                "--generators",      *self._get_generator_sequence(),
                "--chaos-depth",     str(self.chaos_depth.get()),
                "--gendy-breakpoints", str(self.gendy_breakpoints.get()),
                "--gendy-amp-step",  str(self.gendy_amp_step.get()),
                "--gendy-dur-step",  str(self.gendy_dur_step.get()),
                "--gendy-distribution", self.gendy_distribution.get(),
            ]
            if not self.connected.get():
                cmd.append("--no-connected")

            seed_str = self.seed.get().strip()
            if seed_str.isdigit():
                cmd += ["--seed", seed_str]

        if save_insitunit:
            cmd += ["--save-insitunit", save_insitunit]

        if not no_variations:
            cmd += [
                "--num-variations",   str(self.num_variations.get()),
                "--variation-depth",  str(self.variation_depth.get()),
                "--variation-scope",  self.variation_scope.get(),
                "--variation-method", self.variation_method.get(),
            ]

            if self.use_per_param_var.get():
                cfg = self._per_param_var_frame.get_config()
                tmp_pp = "temp_perp_var.json"
                with open(tmp_pp, "w") as f:
                    json.dump(cfg, f)
                cmd += ["--per-param-config", tmp_pp]

        cmd += ["--output-dir", self.output_dir.get()]

        if self.no_synthesize.get():
            cmd.append("--no-synthesize")
        if self.embed_markers.get():
            cmd.append("--embed-markers")

        return cmd

    # ── Logging helper ───────────────────────────────────────────────────────

    def _log_msg(self, msg: str):
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self.update_idletasks()

    # ── Main run ─────────────────────────────────────────────────────────────

    def _run(self):
        if not os.path.exists(SCRIPT_PATH):
            messagebox.showerror("Script not found",
                                 f"Backend script not found:\n{SCRIPT_PATH}")
            return

        cmd = self._build_cmd()
        self._log.delete("1.0", "end")
        self._log_msg("Command:\n  " + " ".join(cmd))
        self._log_msg("─" * 60)

        def worker():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                for line in proc.stdout:
                    self.after(0, self._log_msg, line.rstrip())
                proc.wait()
                if proc.returncode == 0:
                    self.after(0, self._log_msg, "\n✓ Done.")
                else:
                    self.after(0, self._log_msg,
                               f"\n✗ Process exited with code {proc.returncode}")
            except Exception as e:
                self.after(0, messagebox.showerror, "Run error", str(e))

        threading.Thread(target=worker, daemon=True).start()


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = DirectSynthGUI()
    app.mainloop()
