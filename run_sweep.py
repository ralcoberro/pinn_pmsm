"""
FEMM Torque Sweep with Fourier Harmonics
=========================================
Processes all valid FEM models from designs.csv:
  1. Finds the optimal commutation angle (iq_angle_optimo) using the first valid model.
  2. Sweeps 360 electrical degrees for every valid model.
  3. Computes FFT of each torque waveform and stores the first 10 harmonics.

Outputs:
  - dataset/torque_results.csv       (one row per model, harmonics + mean torque)
  - dataset/torque_waveforms/modelXXXX.csv  (full waveform per model)
"""

import os
import sys
import math
import csv
import time
import numpy as np
import femm

# ── import helpers from same directory ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from femm_utils import (
    dq_to_abc,
    load_model,
    solve,
    set_sliding_band_angle,
    get_torque_slidingband,
    set_currents,
)

# ===================================================================
# USER CONFIG
# ===================================================================
DATASET_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
CSV_PATH      = os.path.join(DATASET_DIR, "designs.csv")
DESIGNS_DIR   = os.path.join(DATASET_DIR, "designs")
WAVEFORM_DIR  = os.path.join(DATASET_DIR, "torque_waveforms")
RESULTS_PATH  = os.path.join(DATASET_DIR, "torque_results.csv")

CIRCUITS   = {"A": "A", "B": "B", "C": "C"}
BAND_NAME  = "SlidingBand"

N_HARMONICS     = 10          # DC + harmonics 1‥9
SWEEP_SAMPLES   = 72          # configurable: points per electrical cycle
ANGLE_SEARCH_SAMPLES = 72     # points for the iq‑angle search

# ===================================================================
# Helpers
# ===================================================================

def read_designs(csv_path: str) -> list[dict]:
    """Read designs.csv and return rows as list of dicts (only valid models)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip models with errors or without a FEM file
            if int(row["error_code"]) != 0:
                continue
            fem_file = row.get("fem_file", "").strip()
            if not fem_file:
                continue
            rows.append(row)
    return rows


def find_iq_angle(fem_path: str, pp: int, iq: float,
                  n_steps: int = 72) -> float:
    """
    Find the rotor mechanical angle that maximises torque for id=0, iq=const.

    Sweeps one full electrical cycle (360/pp mechanical degrees) with fixed
    stator current (theta_e=0) and returns the band position of max torque.
    """
    mech_range = 360.0 / pp          # one electrical cycle in mech degrees
    delta = mech_range / n_steps

    load_model(fem_path)

    # Fix current at theta_e = 0
    ia, ib, ic = dq_to_abc(0.0, iq, 0.0)
    set_currents(CIRCUITS, ia, ib, ic)

    # ── Evaluate at angle = 0 (baseline) ────────────────────────
    set_sliding_band_angle(BAND_NAME, 0.0)
    solve()
    T0 = get_torque_slidingband(BAND_NAME)

    best_angle = 0.0
    best_abs_T = abs(T0)
    best_T_signed = T0

    direction = 1        # +1 forward, -1 backward
    angle = 0.0

    # ── Hill-climb on |T| ─────────────────────────────────────
    for k in range(1, n_steps):
        angle += direction * delta

        set_sliding_band_angle(BAND_NAME, angle)
        solve()
        T = get_torque_slidingband(BAND_NAME)

        abs_T = abs(T)
        if abs_T > best_abs_T:
            best_abs_T = abs_T
            best_T_signed = T
            best_angle = angle
        else:
            # |T| is declining
            if k == 1:
                # Wrong direction – reverse and restart from origin
                direction = -direction
                angle = 0.0
            else:
                # Already reversed – we passed the peak, stop
                break

    # ── Handle negative maximum ────────────────────────────────
    if best_T_signed < 0:
        best_T_signed = -best_T_signed
        best_angle += 360.0 / pp

    femm.mi_close()
    print(f"  Optimal iq angle: {best_angle:.4f} mech deg  "
          f"(torque = {best_T_signed:.4f} N·m)")
    return best_angle


def sweep_torque(fem_path: str, pp: int, iq: float,
                 iq_angle: float, n_steps: int = 72) -> dict:
    """
    Sweep one electrical cycle synchronously and return the torque waveform.

    Returns dict with keys: theta_e_deg[], band_pos_deg[], torque[].
    """
    mech_range = 360.0 / pp
    delta_mech = mech_range / n_steps

    load_model(fem_path)

    theta_e_list = []
    band_pos_list = []
    torque_list = []

    for k in range(n_steps):
        band_pos = iq_angle + k * delta_mech
        theta_e  = pp * k * delta_mech * math.pi / 180.0   # rad

        ia, ib, ic = dq_to_abc(0.0, iq, theta_e)
        set_currents(CIRCUITS, ia, ib, ic)
        set_sliding_band_angle(BAND_NAME, band_pos)

        solve()
        T = get_torque_slidingband(BAND_NAME)

        theta_e_list.append(k * 360.0 / n_steps)   # electrical degrees
        band_pos_list.append(band_pos)
        torque_list.append(T)

    femm.mi_close()

    return {
        "theta_e_deg": theta_e_list,
        "band_pos_deg": band_pos_list,
        "torque": torque_list,
    }


def torque_harmonics(torque: list[float], n_harmonics: int = 10) -> tuple:
    """
    Compute FFT of the torque waveform.

    Returns (amplitudes[0..n_harmonics-1], phases_deg[0..n_harmonics-1]).
    Amplitude[0] is the DC component (mean torque).
    """
    arr = np.array(torque)
    N = len(arr)
    spectrum = np.fft.rfft(arr)

    amplitudes = np.abs(spectrum) / N
    amplitudes[0] = amplitudes[0]        # DC component (mean)
    amplitudes[1:] = amplitudes[1:] * 2  # single‑sided scaling

    phases = np.angle(spectrum, deg=True)

    # Pad if fewer freq bins than requested
    n = min(n_harmonics, len(amplitudes))
    amp_out   = list(amplitudes[:n])  + [0.0] * (n_harmonics - n)
    phase_out = list(phases[:n])      + [0.0] * (n_harmonics - n)

    return amp_out, phase_out


def save_waveform(out_path: str, waveform: dict) -> None:
    """Save a single torque waveform to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "theta_e_deg", "band_pos_deg", "torque"])
        for k in range(len(waveform["torque"])):
            w.writerow([
                k,
                f"{waveform['theta_e_deg'][k]:.4f}",
                f"{waveform['band_pos_deg'][k]:.4f}",
                f"{waveform['torque'][k]:.6f}",
            ])


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()

    # ── Read designs.csv ──────────────────────────────────────────
    designs = read_designs(CSV_PATH)
    print(f"Found {len(designs)} valid models in {CSV_PATH}")
    if not designs:
        print("No valid models to process. Exiting.")
        return

    # Common parameters (same for all models in this dataset)
    pp = int(designs[0]["pp"])

    # ── Open FEMM ─────────────────────────────────────────────────
    femm.openfemm(1)          # 1 = minimised

    # ── Step 1: find optimal iq angle from first model ────────────
    first = designs[0]
    first_fem = os.path.join(DESIGNS_DIR, first["fem_file"])
    first_iq  = float(first["phase_current"])

    print(f"\n── Finding optimal iq angle using {first['fem_file']} "
          f"(iq = {first_iq:.2f} A) ──")
    iq_angle_optimo = find_iq_angle(first_fem, pp, first_iq,
                                     n_steps=ANGLE_SEARCH_SAMPLES)

    # ── Step 2 & 3: sweep each model ─────────────────────────────
    results_rows = []
    os.makedirs(WAVEFORM_DIR, exist_ok=True)

    for idx, row in enumerate(designs):
        sample_id = int(row["sample_id"])
        fem_file  = row["fem_file"]
        fem_path  = os.path.join(DESIGNS_DIR, fem_file)
        iq_val    = float(row["phase_current"])

        if not os.path.isfile(fem_path):
            print(f"  [{idx+1}/{len(designs)}] {fem_file} — FILE NOT FOUND, skipping")
            continue

        print(f"  [{idx+1}/{len(designs)}] {fem_file}  (iq = {iq_val:.2f} A) ...",
              end="", flush=True)

        try:
            waveform = sweep_torque(fem_path, pp, iq_val, iq_angle_optimo,
                                    n_steps=SWEEP_SAMPLES)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        # Fourier analysis
        amps, phases = torque_harmonics(waveform["torque"], N_HARMONICS)

        # Save individual waveform
        wf_name = fem_file.replace(".fem", ".csv")
        save_waveform(os.path.join(WAVEFORM_DIR, wf_name), waveform)

        # Build results row
        res = {
            "sample_id": sample_id,
            "iq_angle_optimo": f"{iq_angle_optimo:.4f}",
            "iq": f"{iq_val:.4f}",
            "torque_mean": f"{amps[0]:.6f}",
        }
        for h in range(N_HARMONICS):
            res[f"h{h}_amp"]   = f"{amps[h]:.6f}"
            res[f"h{h}_phase"] = f"{phases[h]:.4f}"

        results_rows.append(res)

        print(f"  T_mean={amps[0]:.4f} N·m")

    # ── Step 4: write summary CSV ─────────────────────────────────
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(RESULTS_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\nResults saved to {RESULTS_PATH}")
        print(f"Waveforms saved to {WAVEFORM_DIR}/")

    # ── Cleanup ───────────────────────────────────────────────────
    femm.closefemm()

    elapsed = time.time() - t0
    print(f"\nDone. Processed {len(results_rows)} models in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
