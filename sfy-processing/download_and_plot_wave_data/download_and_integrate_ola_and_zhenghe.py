#!/usr/bin/env python3
"""Download SFY (zhenghe) wave data AND load a co-located OLA recording, integrate
both to velocity & displacement, and overlay them.

This is download_and_integrate_zhenghe.py + the OLA buoy. The SFY data is fetched
straight from the data-hub with `sfydata axl ts` into a (cached) netCDF; the OLA
data is decoded from a BOOT_* folder of DATA_BOOT_*.dat files, run through the
Mahony vertical-motion AHRS, and time-corrected onto true GNSS UTC. Both are then
double-integrated with the same drift-removal method and plotted on a shared time
axis (OLA blue, SFY red).

Pipeline:
  1. Download the SFY buoy's axl timeseries for [WINDOW_START, WINDOW_END] into a
     netCDF (cached). w_z is world-frame vertical accel INCLUDING gravity; we
     subtract the window mean to get AC vertical accel (+up, gravity removed).
  2. Decode the OLA BOOT_* folder, build a GNSS-micros -> UTC mapping, run the
     Mahony AHRS (-> world-vertical accel, gravity removed), and map the AHRS
     output onto true UTC (the MCU clock is ~450 ppm fast, so the elapsed time
     is scaled by the GNSS-fit slope — the "#4 timebase fix").
  3. Double-integrate accel -> velocity -> displacement for each buoy, with one
     drift-removal method (Butterworth high-pass or savgol drift subtraction).
  4. Plot accel / velocity / displacement, OLA vs SFY overlaid.

REQUIREMENTS (this script straddles two projects):
  * `sfydata` must be on PATH with SFY_SERVER / SFY_READ_TOKEN set (same as the
    other download_*_zhenghe.py scripts — usually via the sfy-processing .env).
  * The OLA decoder must be importable: set OLA_DECODER_DIR below. Its deps
    (numpy, scipy, xarray, loguru) must be installed in the running interpreter.
"""
import sys
import subprocess
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

# Comparison window (UTC). SFY data is downloaded and clipped to this; the OLA
# AHRS runs on this window (+ a short lead-in) and the output is clipped to it.
WINDOW_START = np.datetime64("2026-06-14T20:29:50")
WINDOW_END   = np.datetime64("2026-06-14T20:35:20")

# --- SFY (downloaded) ---
DEV    = "UIO_MEK_ZHENGHE"          # buoy device, as listed by `sfydata list`
NETCDF = HERE / "zhenghe_window.nc"  # cached download (delete / FORCE to refetch)
FORCE_DOWNLOAD = False
FREQ = 52.0                          # keep packages within 2 Hz of this; None to disable

# --- OLA (local BOOT_* folder) ---
# Folder of DATA_BOOT_*.dat files to decode. Must overlap the UTC window.
OLA_BOOT_FOLDER = Path(
    r"C:/projects/2026_ola_logger_imu_gps/decoder/sfy_comparison4/BOOT_000008")
# Where the OLA decoder lives (provides `decoder` and `ahrs_vertical`).
OLA_DECODER_DIR = Path(r"C:/projects/2026_ola_logger_imu_gps/decoder")
MAHONY_KP   = 8.0      # Mahony proportional gain (attitude pull toward gravity)
MAHONY_KI   = 1.25     # Mahony integral gain (online gyro-bias learning)
WINDOW_PAD_S = 5.0     # AHRS lead-in before WINDOW_START (gravity/bias bootstrap)

# Drift-removal method for the double integration. Pick ONE:
#   "butterworth" — zero-phase Butterworth high-pass at each stage. Clean in the
#                   wave band, but rings at its cutoff on transients/spikes.
#   "savgol"      — subtract a long, low-order Savitzky-Golay smooth (slow-drift
#                   estimate) at each stage. Transient-friendly, no ringing.
METHOD = "butterworth"

_METHOD_KEY = {"butterworth": "A_butterworth_hp", "savgol": "B_savgol_detrend"}
if METHOD not in _METHOD_KEY:
    raise SystemExit(f"METHOD must be one of {list(_METHOD_KEY)}, got {METHOD!r}")

# -------------------- integration pipeline (self-contained) ------------------
HP_ORDER = 4                      # Butterworth order (zero-phase via filtfilt)
SAVGOL_WIN_S = 8.0                # savgol smoothing window (s); ~1/win is the corner
SAVGOL_ORDER = 2                  # savgol polynomial order (low = "hard" smoothing)

METHODS = {
    "A_butterworth_hp": dict(
        accel_detrend=True,  accel_hp_hz=None,
        vel_detrend=False,   vel_hp_hz=0.05,
        disp_detrend=False,  disp_hp_hz=0.2,
    ),
    "B_savgol_detrend": dict(
        accel_detrend=True,
        vel_savgol_s=SAVGOL_WIN_S,
        disp_savgol_s=SAVGOL_WIN_S,
    ),
}


def _clean(x, fs, do_detrend=False, hp_hz=None, savgol_s=None,
           savgol_order=SAVGOL_ORDER, hp_order=HP_ORDER):
    """Optional linear-detrend, then savgol-drift subtraction, then zero-phase
    Butterworth high-pass (any subset, in that order)."""
    if do_detrend:
        x = detrend(x, type="linear")
    if savgol_s is not None:
        win = int(round(savgol_s * fs))
        win += 1 - (win % 2)                       # force odd
        win = min(win, len(x) - (1 - len(x) % 2))  # keep < len, odd
        if win > savgol_order:
            x = x - savgol_filter(x, win, savgol_order)
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
    """accel (m/s², +up) on an int64-ns grid -> (utc, accel_raw, accel, vel, disp, fs)."""
    t = (t_ns - t_ns[0]) / 1e9
    fs = 1.0 / np.median(np.diff(t))
    tu = np.arange(0.0, t[-1], 1.0 / fs)
    au_raw = np.interp(tu, t, accel)
    dt = 1.0 / fs

    au = _clean(au_raw, fs, accel_detrend, accel_hp_hz, accel_savgol_s, savgol_order, hp_order)
    vel = cumtrapz(au, dx=dt, initial=0.0)
    vel = _clean(vel, fs, vel_detrend, vel_hp_hz, vel_savgol_s, savgol_order, hp_order)
    disp = cumtrapz(vel, dx=dt, initial=0.0)
    disp = _clean(disp, fs, disp_detrend, disp_hp_hz, disp_savgol_s, savgol_order, hp_order)
    utc = (t_ns[0] + (tu * 1e9).astype(np.int64)).astype("datetime64[ns]")
    return utc, au_raw, au, vel, disp, fs


# ------------------------------- SFY download --------------------------------


def download_window(dev: str, start, end, out: Path, freq=None) -> None:
    """Fetch `dev`'s axl timeseries clipped to [start, end] into `out` via
    `sfydata axl ts`."""
    tx_start = (start - np.timedelta64(6, "h"))
    cmd = [
        "sfydata", "axl", "ts", dev,
        "--tx-start", str(tx_start.astype("datetime64[s]")),
        "--start",    str(start.astype("datetime64[s]")),
        "--end",      str(end.astype("datetime64[s]")),
        "--file",     str(out),
    ]
    if freq is not None:
        cmd += ["--freq", str(freq)]
    print("Downloading:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "`sfydata` not found on PATH. Activate the sfy-processing environment "
            "so SFY_SERVER / SFY_READ_TOKEN and the CLI are available.")
    if not out.exists():
        raise RuntimeError(f"sfydata produced no file at {out} — no data in window?")


# ------------------------------- OLA loading ---------------------------------
# Imported lazily from the OLA decoder project (see OLA_DECODER_DIR).

def _import_ola_decoder():
    if str(OLA_DECODER_DIR) not in sys.path:
        sys.path.insert(0, str(OLA_DECODER_DIR))
    try:
        from decoder import decode_file, load_data_as_arrays  # noqa
        import ahrs_vertical  # noqa
    except Exception as exc:
        raise SystemExit(
            f"Could not import the OLA decoder from {OLA_DECODER_DIR}.\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "Set OLA_DECODER_DIR correctly and run with an interpreter that has "
            "the decoder deps (numpy, scipy, xarray, loguru) installed.")
    try:                                  # keep the AHRS library quiet on stderr
        from loguru import logger as _L
        _L.remove()
    except Exception:
        pass
    return decode_file, load_data_as_arrays, ahrs_vertical


def _load_ola_combined(folder: Path, decode_file, load_data_as_arrays) -> dict:
    """Decode every DATA_BOOT_*.dat in the folder, concatenate, return one dict."""
    files = sorted(folder.glob("DATA_BOOT_*.dat"))
    if not files:
        raise FileNotFoundError(f"No DATA_BOOT_*.dat under {folder}")
    per_file = [load_data_as_arrays(decode_file(f, allow_no_pps=True)["file"]) for f in files]
    combined: dict = {}
    for k in per_file[0].keys():
        v0 = per_file[0][k]
        if not isinstance(v0, np.ndarray):
            combined[k] = v0
            continue
        try:
            combined[k] = np.concatenate(
                [d[k] for d in per_file if isinstance(d[k], np.ndarray)])
        except ValueError:
            combined[k] = v0
    return combined


def _build_utc_mapping(data: dict) -> tuple:
    """Return (slope, intercept) so utc_posix = slope*micros + intercept, fit over
    the GNSS PVT entries with a plausible posix (skips the firmware drift spike)."""
    g_micros = np.asarray(data.get("gnss_micros_unwrapped", []), dtype=np.float64)
    g_posix  = np.asarray(data.get("gnss_posix", []), dtype=np.float64)
    if g_micros.size < 2 or g_micros.size != g_posix.size:
        raise RuntimeError("OLA recording lacks GNSS posix data for UTC mapping")
    valid = g_posix > 1e9
    if valid.sum() < 2:
        raise RuntimeError("No usable GNSS fixes in the OLA recording")
    slope, intercept = np.polyfit(g_micros[valid], g_posix[valid], 1)
    return float(slope), float(intercept)


def load_ola_vertical(folder, start, end, pad_s, kp, ki):
    """Decode + AHRS the OLA folder, return (t_ns, accel_up) on true UTC, clipped
    to [start, end]. accel_up is gravity-removed world-vertical accel (m/s²)."""
    decode_file, load_data_as_arrays, ahrs = _import_ola_decoder()
    ola = _load_ola_combined(folder, decode_file, load_data_as_arrays)
    n = len(ola["imu_micros_unwrapped"])
    slope, intercept = _build_utc_mapping(ola)
    im = np.asarray(ola["imu_micros_unwrapped"], dtype=np.float64)
    utc = slope * im + intercept                         # true UTC posix seconds

    A = start.astype("datetime64[ns]").astype("int64") / 1e9
    B = end.astype("datetime64[ns]").astype("int64") / 1e9
    sel = (utc >= A - pad_s) & (utc <= B)                # window + lead-in
    if sel.sum() < 100:
        raise SystemExit(
            f"Only {int(sel.sum())} OLA samples in the window — does "
            f"{folder.name} overlap {start}..{end}?")
    w = dict(ola)
    for k, v in ola.items():
        if isinstance(v, np.ndarray) and v.size == n:
            w[k] = v[sel]

    res = ahrs.compute_vertical_motion_mahony(
        w, kp=kp, ki=ki, motion_gate_threshold=1e9, calibrate_gyro_bias=False)

    # #4 timebase fix: res.t is MCU-clock seconds (Artemis ~+450 ppm fast); scale
    # the elapsed time by the GNSS slope so it shares the SFY/true UTC clock.
    utc0 = utc[sel][0]
    t_utc = utc0 + res.t * (slope / 1e-6)
    m = (t_utc >= A) & (t_utc <= B)                      # clip to the window
    t_ns = (t_utc[m] * 1e9).astype(np.int64)
    accel = res.accel_z_up_raw[m]
    return t_ns, accel


# ------------------------------- main ----------------------------------------


def main() -> None:
    cfg = METHODS[_METHOD_KEY[METHOD]]

    # === SFY: download + load ===
    if FORCE_DOWNLOAD or not NETCDF.exists():
        download_window(DEV, WINDOW_START, WINDOW_END, NETCDF, freq=FREQ)
    else:
        print(f"Using cached {NETCDF.name} (set FORCE_DOWNLOAD=True to refetch)")

    print(f"\nLoading SFY -> {NETCDF}")
    sfy = xr.open_dataset(NETCDF).sel(time=slice(WINDOW_START, WINDOW_END))
    sfy_t = sfy.time.values
    if sfy_t.size == 0:
        raise SystemExit("No SFY samples in the window — check WINDOW_START/END.")
    sfy_wz = sfy.w_z.values.astype(np.float64)
    sfy_accel = sfy_wz - float(np.mean(sfy_wz))          # gravity removed (AC)
    sfy_t_ns = sfy_t.astype("datetime64[ns]").astype(np.int64)
    print(f"  SFY: {sfy_t.size} samples, w_z mean={np.mean(sfy_wz):.3f} m/s², "
          f"AC std={np.std(sfy_accel):.3f} m/s²")

    # === OLA: decode + AHRS + time-correct ===
    print(f"\nLoading OLA -> {OLA_BOOT_FOLDER}")
    ola_t_ns, ola_accel = load_ola_vertical(
        OLA_BOOT_FOLDER, WINDOW_START, WINDOW_END, WINDOW_PAD_S, MAHONY_KP, MAHONY_KI)
    print(f"  OLA: {ola_t_ns.size} samples, AC std={np.std(ola_accel):.3f} m/s²")

    # === Integrate both with the selected method ===
    buoys = {}
    for name, t_ns, accel in (("OLA", ola_t_ns, ola_accel),
                              ("SFY", sfy_t_ns, sfy_accel)):
        utc, au_raw, au, vel, disp, fs = integrate_to_displacement(t_ns, accel, **cfg)
        buoys[name] = dict(utc=utc, accel_raw=au_raw, accel=au, vel=vel, disp=disp, fs=fs)
        print(f"  {name}: fs={fs:6.2f} Hz  vel std={vel.std()*100:5.1f} cm/s  "
              f"disp std={disp.std()*100:5.1f} cm  (Hs~4*std={4*disp.std()*100:5.1f} cm)")

    # === Plot: accel / velocity / displacement, OLA vs SFY ===
    bcol = {"OLA": "tab:blue", "SFY": "tab:red"}
    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
    for name, o in buoys.items():
        ax[0].plot(o["utc"], o["accel"],       bcol[name], lw=0.7, label=name)
        ax[1].plot(o["utc"], o["vel"] * 100,   bcol[name], lw=0.7, label=name)
        ax[2].plot(o["utc"], o["disp"] * 100,  bcol[name], lw=0.8, label=name)
    ax[0].set_ylabel("accel +up (m/s²)")
    ax[1].set_ylabel("velocity (cm/s)")
    ax[2].set_ylabel("displacement (cm)")
    ax[2].set_xlabel("UTC")
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # shared, symmetric y-limits per row from both buoys
    for row, key, scale in ((0, "accel", 1.0), (1, "vel", 100.0), (2, "disp", 100.0)):
        lim = 1.2 * max(np.max(np.abs(o[key] * scale)) for o in buoys.values())
        ax[row].set_ylim(-lim, lim)
    for a_ in ax:
        a_.grid(True, alpha=0.3)
        a_.legend(loc="upper right", fontsize=8)

    method_desc = ("Butterworth high-pass" if METHOD == "butterworth"
                   else "savgol drift subtraction")
    fig.suptitle(
        f"OLA ({OLA_BOOT_FOLDER.name}) vs SFY ({DEV}) — accel -> velocity -> displacement\n"
        f"window {WINDOW_START}..{WINDOW_END}   method: {method_desc}")
    fig.tight_layout()

    out_png = HERE / f"ola_zhenghe_integrated_displacement_{METHOD}.png"
    fig.savefig(out_png, dpi=90)
    print(f"\nSaved {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
