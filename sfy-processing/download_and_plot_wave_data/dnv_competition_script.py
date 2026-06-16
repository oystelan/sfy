#!/usr/bin/env python3
"""Download SFY (zhenghe) vertical-acceleration data for a fixed UTC window,
integrate it to velocity and displacement, and plot the three stages.

This is the SFY-only counterpart to integrate_displacement.py — all OLA-related
machinery has been removed. The data is fetched straight from the data-hub with
`sfydata axl ts` (the same tool download_data_from_zhenghe.py uses), clipped to
[WINDOW_START, WINDOW_END], then loaded with xarray.

Pipeline:
  1. Download the buoy's axl timeseries for the window into a (cached) netCDF.
     Variable `w_z` is the world-frame vertical acceleration in m/s² INCLUDING
     gravity (its window mean is ~9.5-9.8 m/s²). Subtract the window mean to get
     the AC vertical acceleration (+up, gravity removed).
  2. Double-integrate accel -> velocity -> displacement. Double integration
     amplifies any low-frequency residual by 1/w², so each stage needs a
     drift-removal step. Two approaches are compared (imported verbatim from
     integrate_displacement.py):
       A_butterworth_hp — zero-phase Butterworth high-pass at each stage. Clean
           in the wave band, but rings at its cutoff on transients.
       B_savgol_detrend — subtract a long, low-order Savitzky-Golay smooth (the
           slow-drift estimate) at each stage. Transient-friendly, no ringing.
  3. Plot accel / velocity / displacement, methods A vs B overlaid.

Requires the SFY_SERVER and SFY_READ_TOKEN environment variables (as the other
sfydata commands do; usually provided via the .env / direnv in sfy-processing).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from scipy.signal import butter, filtfilt, detrend, savgol_filter
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:                       # older scipy
    from scipy.integrate import cumtrapz as _ct
    cumtrapz = lambda y, dx, initial=0: _ct(y, dx=dx, initial=initial)

HERE = Path(__file__).parent

# ----------------------------- user knobs ------------------------------------

# Comparison window (UTC). Data is downloaded and clipped to this interval.
WINDOW_START = np.datetime64("2026-06-14T20:29:50")
WINDOW_END   = np.datetime64("2026-06-14T20:35:20")

# Buoy device (as listed by `sfydata list`).
DEV = "UIO_MEK_ZHENGHE"

# Where the downloaded netCDF is cached. Delete it (or set FORCE_DOWNLOAD=True)
# to re-fetch from the hub.
NETCDF = HERE / "zhenghe_dnv.nc"
FORCE_DOWNLOAD = False

# list of compentitors and when they jumped
competitordata = [["jorn", "2026-06-14T20:30:50"],
                  ["magnus", "2026-06-14T20:31:44"],
                  ["odin", "2026-06-14T20:33:02"]]

# Only keep packages whose advertised frequency is within 2 Hz of this (the SFY
# runs at ~52 Hz here). Set to None to disable the frequency filter.
FREQ = 52.0

# Drift-removal method for the double integration. Pick ONE:
#   "butterworth" — zero-phase Butterworth high-pass at each stage. Clean in the
#                   wave band, but rings at its cutoff on transients/spikes.
#   "savgol"      — subtract a long, low-order Savitzky-Golay smooth (the slow-
#                   drift estimate) at each stage. Transient-friendly, no ringing.
METHOD = "butterworth"

# Map the switch to the corresponding integration config in METHODS (below).
_METHOD_KEY = {"butterworth": "A_butterworth_hp", "savgol": "B_savgol_detrend"}
if METHOD not in _METHOD_KEY:
    raise SystemExit(f"METHOD must be one of {list(_METHOD_KEY)}, got {METHOD!r}")


# -------------------- integration pipeline (self-contained) ------------------
# Double-integrating acceleration amplifies any low-frequency residual by 1/w²,
# so a raw cumulative integral drifts away. Each integration stage therefore
# needs a drift-removal step. Two approaches are provided:
#
#   A_butterworth_hp — high-pass (zero-phase Butterworth) the velocity and the
#       displacement. Clean in the wave band, but a Butterworth's impulse
#       response RINGS at its cutoff, so any transient/spike in the record turns
#       into a large sinusoidal bump at ~HP_HZ that can dominate the result.
#
#   B_savgol_detrend — estimate the slow drift with a long, low-order Savitzky-
#       Golay smooth and SUBTRACT it (at the velocity and displacement stages).
#       The smooth barely follows a localized transient, so the transient
#       survives the subtraction roughly intact — no ringing bump — while the
#       slow drift is still removed. SAVGOL_WIN_S sets the window (~1/win is the
#       effective corner); a low SAVGOL_ORDER makes it a "hard" smooth.

HP_ORDER = 4                      # Butterworth order (zero-phase via filtfilt)
SAVGOL_WIN_S = 8.0                # savgol smoothing window (s); ~1/win is the corner
SAVGOL_ORDER = 2                  # savgol polynomial order (low = "hard" smoothing)

# Each integration stage can independently DETREND (remove linear trend),
# HIGH-PASS (Butterworth, zero-phase), and/or SAVGOL-DETREND (subtract a savgol
# smooth). Pass the relevant *_hp_hz / *_savgol_s; leave None to skip.
METHODS = {
    "A_butterworth_hp": dict(
        accel_detrend=True,  accel_hp_hz=None,
        vel_detrend=False,   vel_hp_hz=0.05,
        disp_detrend=False,  disp_hp_hz=0.2,
    ),
    "B_savgol_detrend": dict(
        accel_detrend=True,                         # 1) basic accel detrend (as A)
        vel_savgol_s=SAVGOL_WIN_S,                  # 2) subtract savgol drift from velocity
        disp_savgol_s=SAVGOL_WIN_S,                 # 3) subtract savgol drift from displacement
    ),
}


def _clean(x, fs, do_detrend=False, hp_hz=None, savgol_s=None,
           savgol_order=SAVGOL_ORDER, hp_order=HP_ORDER):
    """Clean a signal: optional linear-detrend, then savgol-drift subtraction,
    then zero-phase Butterworth high-pass (any subset, in that order)."""
    if do_detrend:
        x = detrend(x, type="linear")
    if savgol_s is not None:
        win = int(round(savgol_s * fs))
        win += 1 - (win % 2)                       # force odd
        win = min(win, len(x) - (1 - len(x) % 2))  # keep < len, odd
        if win > savgol_order:
            x = x - savgol_filter(x, win, savgol_order)   # subtract the drift estimate
    if hp_hz is not None:
        b, a = butter(hp_order, hp_hz / (fs / 2.0), btype="highpass")
        x = filtfilt(b, a, x)
    return x


def integrate_to_displacement(
    t_ns, accel,
    accel_detrend=True, accel_hp_hz=None, accel_savgol_s=None,
    vel_detrend=False,  vel_hp_hz=None,   vel_savgol_s=None,
    disp_detrend=False, disp_hp_hz=None,  disp_savgol_s=None,
    savgol_order=SAVGOL_ORDER, hp_order=HP_ORDER,
):
    """accel (m/s², +up) on an int64-ns grid -> (tu, accel_raw, accel, vel, disp, fs).

    Resamples to a uniform grid at the median rate (so the zero-phase filters are
    valid), then integrates twice. Each stage (accel, velocity, displacement) is
    independently detrended, savgol-drift-subtracted, and/or high-passed.
    """
    t = (t_ns - t_ns[0]) / 1e9                 # seconds from start
    fs = 1.0 / np.median(np.diff(t))
    tu = np.arange(0.0, t[-1], 1.0 / fs)       # uniform grid
    au_raw = np.interp(tu, t, accel)           # resampled, BEFORE any cleaning
    dt = 1.0 / fs

    au = _clean(au_raw, fs, accel_detrend, accel_hp_hz, accel_savgol_s, savgol_order, hp_order)
    vel = cumtrapz(au, dx=dt, initial=0.0)
    vel = _clean(vel, fs, vel_detrend, vel_hp_hz, vel_savgol_s, savgol_order, hp_order)
    disp = cumtrapz(vel, dx=dt, initial=0.0)
    disp = _clean(disp, fs, disp_detrend, disp_hp_hz, disp_savgol_s, savgol_order, hp_order)
    return tu, au_raw, au, vel, disp, fs


# ------------------------------- download ------------------------------------


def download_window(dev: str, start, end, out: Path, freq=None) -> None:
    """Fetch `dev`'s axl timeseries clipped to [start, end] into `out` (netCDF)
    via `sfydata axl ts`. A small tx-time margin is added so transmission delay
    doesn't drop packages whose data falls inside the window."""
    # Invoke the sfydata CLI through the SAME interpreter that runs this script
    # (`python -m sfy.cli.sfydata`) rather than a bare `sfydata` on PATH — when
    # run from VSCode's Run button the conda env's Scripts dir often isn't on
    # PATH, but the interpreter itself is the sfy env (its xarray/scipy imported
    # above), so the module is importable and this just works.
    # sfydata accepts click.DateTime() formats; str(np.datetime64) -> ISO 8601.
    tx_start = (start - np.timedelta64(6, "h"))  # widen the tx search backwards
    cmd = [
        sys.executable, "-m", "sfy.cli.sfydata", "axl", "ts", dev,
        "--tx-start", str(tx_start.astype("datetime64[s]")),
        "--start",    str(start.astype("datetime64[s]")),
        "--end",      str(end.astype("datetime64[s]")),
        "--file",     str(out),
    ]
    if freq is not None:
        cmd += ["--freq", str(freq)]
    print("Downloading:", " ".join(cmd))
    try:
        # Run from the sfy-processing root so `sfy` is importable even if the
        # package is only present as source (not pip-installed) in this env.
        # `out` is an absolute path, so the cwd change doesn't affect it.
        subprocess.run(cmd, check=True, cwd=str(HERE.parent))
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"sfydata failed (exit {e.returncode}). Make sure this script is run "
            "with the sfy-processing environment's Python (the one that runs "
            "download_data_from_zhenghe.py) and that SFY_SERVER / SFY_READ_TOKEN "
            "are set in the environment.")
    if not out.exists():
        raise RuntimeError(
            f"sfydata produced no file at {out} — no data in the window?")


# ------------------------------- main ----------------------------------------


def main() -> None:
    if FORCE_DOWNLOAD or not NETCDF.exists():
        download_window(DEV, WINDOW_START, WINDOW_END, NETCDF, freq=FREQ)
    else:
        print(f"Using cached {NETCDF.name} (set FORCE_DOWNLOAD=True to refetch)")

    # === Load SFY ===
    print(f"\nLoading SFY -> {NETCDF}")
    sfy = xr.open_dataset(NETCDF)
    print(f"  variables : {list(sfy.data_vars)}")
    print(f"  time range: {sfy.time.values[0]} .. {sfy.time.values[-1]}")
    est = sfy.attrs.get("estimated_frequency")
    print(f"  frequency : {sfy.attrs.get('frequency')} Hz"
          + (f" (estimated {est:.2f})" if est is not None else ""))

    sfy_win = sfy.sel(time=slice(WINDOW_START, WINDOW_END))
    sfy_t   = sfy_win.time.values
    sfy_wz  = sfy_win.w_z.values.astype(np.float64)
    if sfy_t.size == 0:
        raise SystemExit("No SFY samples in the window — check WINDOW_START/END.")

    # Remove gravity: w_z averages ~9.5-9.8 m/s². The window is short enough that
    # this is essentially constant, so subtracting the window mean leaves the AC
    # vertical acceleration (+up, gravity removed).
    sfy_wz_ac = sfy_wz - float(np.mean(sfy_wz))
    print(f"  in window : {sfy_t.size} samples, w_z mean={np.mean(sfy_wz):.3f} m/s², "
          f"AC std={np.std(sfy_wz_ac):.3f} m/s²")

    # === Integrate accel -> velocity -> displacement, for each method ===
    t_ns = sfy_t.astype("datetime64[ns]").astype(np.int64)
    mkey = _METHOD_KEY[METHOD]
    cfg = METHODS[mkey]
    tu, au_raw, au, vel, disp, fs = integrate_to_displacement(
        t_ns, sfy_wz_ac, **cfg)
    utc = (t_ns[0] + (tu * 1e9).astype(np.int64)).astype("datetime64[ns]")
    print(f"  {METHOD} ({mkey}): fs={fs:5.2f} Hz  vel std={vel.std()*100:5.1f} cm/s  "
          f"disp std={disp.std()*100:5.1f} cm  (Hs~4*std={4*disp.std()*100:5.1f} cm)")

    # === Plot: accel / velocity / displacement for the selected method ===
    col = "tab:green" if METHOD == "butterworth" else "tab:orange"

    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)

    # Top: raw (pre-cleaning) resampled accel plus the cleaned accel.
    ax[0].plot(utc, au_raw, color="0.6", lw=0.6, label="raw (resampled)")
    ax[0].plot(utc, au,        col, lw=0.7, label=METHOD)
    ax[1].plot(utc, vel * 100, col, lw=0.7, label=METHOD)
    ax[2].plot(utc, disp * 100, col, lw=0.8, label=METHOD)

    ax[0].set_ylabel("accel +up (m/s²)")
    ax[1].set_ylabel("velocity (cm/s)")
    ax[2].set_ylabel("displacement (cm)")
    ax[2].set_xlabel("UTC")
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # Scale accel to the raw signal; velocity/displacement to their own signal.
    ax[0].set_ylim(-1.1 * np.max(np.abs(au_raw)), 1.1 * np.max(np.abs(au_raw)))
    for a_, sig in ((ax[1], vel * 100), (ax[2], disp * 100)):
        lim = 1.3 * np.max(np.abs(sig))
        a_.set_ylim(-lim, lim)

    # === Mark competitor jump times ===
    # A vertical dashed line at each competitor's jump time across all panels,
    # with the name labelled vertically along the line on the top panel.
    for name, tstr in competitordata:
        t = np.datetime64(tstr)
        if not (WINDOW_START <= t <= WINDOW_END):
            print(f"  note: {name} jump {tstr} is outside the window — skipped")
            continue
        for a_ in ax:
            a_.axvline(t, color="tab:red", ls="--", lw=1.0, alpha=0.7)
        # x in data coords, y in axes-fraction so the label always sits at the
        # top of the upper panel regardless of the y-limits.
        ax[0].annotate(name, xy=(t, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(3, -3), textcoords="offset points",
                       rotation=90, va="top", ha="left",
                       fontsize=8, color="tab:red", fontweight="bold")

    for a_ in ax:
        a_.grid(True, alpha=0.3)
        a_.legend(loc="upper right", fontsize=8)
    method_desc = ("Butterworth high-pass" if METHOD == "butterworth"
                   else "savgol drift subtraction")
    fig.suptitle(
        f"SFY {DEV} — accel integrated to velocity & displacement\n"
        f"window {WINDOW_START}..{WINDOW_END}   method: {method_desc}")
    fig.tight_layout()

    out_png = HERE / f"zhenghe_integrated_displacement_{METHOD}.png"
    fig.savefig(out_png, dpi=90)
    print(f"\nSaved {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
