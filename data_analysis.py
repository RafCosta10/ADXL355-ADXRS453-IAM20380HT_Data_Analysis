import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import struct
import time
from typing import Iterable, Optional, Tuple, Union

ArrayLike = Union[np.ndarray, pd.Series]

def mean_center_df(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subtract the column-wise mean from the selected columns in df.
    Returns (centered_df, means_series).
    """
    if cols is None:
        # Use only numeric columns by default
        cols = df.select_dtypes(include=[np.number]).columns

    means = df[cols].mean()
    centered = df.copy()
    centered[cols] = df[cols] - means
    return centered, means

def _cumulative_trapezoid(y: ArrayLike, t: ArrayLike) -> np.ndarray:
    """
    Cumulative trapezoidal integration with non-uniform time steps.
    Returns an array the same length as y, starting with 0.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.shape[0] != t.shape[0]:
        raise ValueError("y and t must have the same length")

    # Compute trapezoids for each interval i -> i+1
    dt = np.diff(t)  # shape (n-1,)
    # average height over each interval
    avg_height = 0.5 * (y[:-1] + y[1:])
    # area of each trapezoid
    area = avg_height * dt  # shape (n-1,)

    # cumulative sum, with initial 0 to align lengths
    cumulative = np.concatenate([[0.0], np.cumsum(area)])
    return cumulative

def double_integrate_trapz(df: pd.DataFrame, t: ArrayLike, cols: Optional[Iterable[str]] = None, interpolate_nans: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform two successive trapezoidal integrations over time for selected columns.

    Args:
        df: DataFrame containing the signals (columns are signals).
        t: 1D array-like of timestamps (same length as df). Can be pd.Series or np.ndarray.
        cols: Columns to integrate. Defaults to all numeric columns.
        interpolate_nans: If True, linearly interpolates NaNs before integrating.

    Returns:
        (first_integral_df, second_integral_df) as DataFrames with the same index/columns as df[cols].
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    t_arr = np.asarray(t, dtype=float)
    if len(t_arr) != len(df):
        raise ValueError("t must have the same number of rows as df")

    # Prepare data (optional NaN handling)
    work = df[cols].astype(float).copy()
    if interpolate_nans:
        # Interpolate within the data; leave leading/trailing NaNs as-is then fill with nearest
        work = work.interpolate(method="linear", limit_direction="both", axis=0)

    # First integration
    first_int = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for c in cols:
        first_int[c] = _cumulative_trapezoid(work[c].to_numpy(), t_arr)

    # Second integration (integrate the result again over t)
    second_int = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for c in cols:
        second_int[c] = _cumulative_trapezoid(first_int[c].to_numpy(), t_arr)

    return first_int, second_int


def mean_center_and_double_integrate(df: pd.DataFrame, t: ArrayLike, cols: Optional[Iterable[str]] = None, interpolate_nans: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper that:
    1) mean-centers selected columns,
    2) returns first and second trapezoidal integrals.

    Returns:
        centered_df, means, first_integral_df, second_integral_df
    """
    centered, means = mean_center_df(df, cols=cols)
    first_int, second_int = double_integrate_trapz(centered, t=t, cols=cols, interpolate_nans=interpolate_nans)
    return centered, means, first_int, second_int


_TS_SCALE = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}

def _extract_ts(buf: memoryview, off: int, ts_type: str, unit: str):
    # ts_type: 'd' (double seconds), 'Q' (uint64 ticks), 'I' (uint32 ticks), 'II' (sec + nsec)
    if ts_type == 'd':
        val = struct.unpack_from('<d', buf, off)[0]
        return float(val)
    elif ts_type == 'Q':
        val = struct.unpack_from('<Q', buf, off)[0]
        return float(val) * _TS_SCALE[unit]
    elif ts_type == 'I':
        val = struct.unpack_from('<I', buf, off)[0]
        return float(val) * _TS_SCALE[unit]
    elif ts_type == 'II':
        sec = struct.unpack_from('<I', buf, off)[0]
        nsec = struct.unpack_from('<I', buf, off + 4)[0]
        return float(sec) + float(nsec) * 1e-9
    else:
        return float('nan')

def _detect_record_format(file_path, candidates, probe_records=1000):
    """
    Detect the record format for a given file.
    candidates: list of dicts like {'name','size','fmt','ts': ('d'|'Q'|'I'|'II', unit)}
    Returns the chosen candidate dict.
    """
    file_size = os.path.getsize(file_path)
    viable = [c for c in candidates if (c.get('size') and file_size % c['size'] == 0 and (file_size // c['size']) >= 2)]
    if not viable:
        return candidates[0]

    now_ts = time.time()
    low_ts = now_ts - 10 * 365 * 24 * 3600.0   # now - 10y
    high_ts = now_ts + 10 * 365 * 24 * 3600.0  # now + 10y

    best = None
    best_score = None

    for c in viable:
        try:
            rec_size = c['size']
            ts_type, unit = c.get('ts', ('d', 's'))
            nrec = min(max(8, probe_records), file_size // rec_size)
            with open(file_path, 'rb') as f:
                buf = f.read(nrec * rec_size)
            mv = memoryview(buf)

            # Extract timestamps according to ts_type
            ts = []
            bad_ii = 0
            for i in range(nrec):
                off = i * rec_size
                if ts_type == 'II':
                    # also validate sec/nsec
                    sec = struct.unpack_from('<I', mv, off)[0]
                    nsec = struct.unpack_from('<I', mv, off + 4)[0]
                    if not (low_ts <= float(sec) <= high_ts) or (nsec >= 1_000_000_000):
                        bad_ii += 1
                    t = float(sec) + float(nsec) * 1e-9
                elif ts_type == 'Q':
                    t = float(struct.unpack_from('<Q', mv, off)[0]) * _TS_SCALE.get(unit, 1.0)
                elif ts_type == 'I':
                    t = float(struct.unpack_from('<I', mv, off)[0]) * _TS_SCALE.get(unit, 1.0)
                else:  # 'd'
                    t = float(struct.unpack_from('<d', mv, off)[0])
                ts.append(t)

            ts = np.asarray(ts, dtype=float)
            diffs = np.diff(ts)
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size == 0:
                continue

            med = float(np.median(diffs))
            neg = int(np.sum(diffs <= 0))
            # dt plausible for high-rate sensors (1 kHz/250 Hz) with slack
            plausible_dt = (1e-6 <= med <= 2e-2)

            # epoch plausibility
            ts_min = float(np.nanmin(ts))
            ts_max = float(np.nanmax(ts))
            epoch_ok = (low_ts <= ts_min <= high_ts and low_ts <= ts_max <= high_ts)

            # penalties
            invalid_pen = 0 if plausible_dt else 2
            epoch_pen = 0 if epoch_ok else 1
            ii_pen = bad_ii  # count of invalid II rows in probe

            # prefer double > Q > II > I
            ts_priority = {'d': 0, 'Q': 1, 'II': 2, 'I': 3}.get(ts_type, 4)

            with np.errstate(all='ignore'):
                jitter = float(np.std(diffs))

            score = (invalid_pen, epoch_pen, ii_pen, ts_priority, neg, jitter)

            if best_score is None or score < best_score:
                best = c
                best_score = score
        except Exception:
            continue

    return best or viable[0]


def _combine_ii_timestamp(rows):
    """
    Convert rows that start with (uint32 sec, uint32 nsec, ...) into
    (float seconds, ...) where seconds = sec + nsec*1e-9.
    """
    out = []
    for r in rows:
        if len(r) < 3:
            continue
        sec, nsec = r[0], r[1]
        ts = float(sec) + float(nsec) * 1e-9
        out.append((ts,) + tuple(r[2:]))
    return out


def _convert_rows_timestamp(rows, ts):
    """
    Normalize first field to float seconds for non-double timestamp formats.
    ts is a tuple like ('d','s'), ('Q','ns'), ('II','ns'), ('I','ms'), etc.
    """
    if not rows:
        return rows
    ts_type, unit = ts
    if ts_type == 'd':
        return rows
    if ts_type == 'II':
        return _combine_ii_timestamp(rows)
    if ts_type in ('Q', 'I'):
        scale = _TS_SCALE.get(unit, 1.0)
        out = []
        for r in rows:
            if not r:
                continue
            t = float(r[0]) * scale
            out.append((t,) + tuple(r[1:]))
        return out
    return rows


def load_and_process_accelerometer_files(folder_path):
    """
    Load and process all .bin files for three accelerometers.
    Supports:
      - Old chunks: {start}_chunk_####.bin with records as 4 doubles (dddd)
      - New single files: accel_1_YYYYMMDD_HHMMSS.bin with records as dddd or dfff
    """
    all_data = {1: [], 2: [], 3: []}
    accel_name_patterns = {
        1: re.compile(r'(accel_?1|accel_1|accl1)', re.IGNORECASE),
        2: re.compile(r'(accel_?2|accel_2|accl2)', re.IGNORECASE),
        3: re.compile(r'(accel_?3|accel_3|accl3)', re.IGNORECASE),
    }
    old_chunk_pat = re.compile(r'(\d+\.\d+)_chunk_\d+\.bin')

    for accel_id in [1, 2, 3]:
        accel_folder = os.path.join(folder_path, f'accl{accel_id}')
        if not os.path.exists(accel_folder):
            print(f"Warning: Accelerometer folder {accel_folder} not found")
            continue

        bin_files = sorted([fn for fn in os.listdir(accel_folder) if fn.lower().endswith('.bin')])
        if not bin_files:
            print(f"Warning: No .bin files found in {accel_folder}")
            continue

        for filename in bin_files:
            file_path = os.path.join(accel_folder, filename)

            is_old = bool(old_chunk_pat.match(filename))
            is_new = bool(re.search(r'accel[_-]?%d' % accel_id, filename, re.IGNORECASE))

            if not (is_old or is_new):
                if len(bin_files) > 1:
                    continue

            # Restrict candidates by naming:
            if is_old:
                candidates = [
                    {'name': 'dddd', 'size': 32, 'fmt': '<dddd', 'ts': ('d', 's')},
                ]
            else:
                candidates = [
                    {'name': 'dddd', 'size': 32, 'fmt': '<dddd', 'ts': ('d', 's')},
                    {'name': 'dfff', 'size': 20, 'fmt': '<dfff', 'ts': ('d', 's')},
                ]

            chosen = _detect_record_format(file_path, candidates)
            rows = _read_binary_records(file_path, chosen['fmt'])
            # 'd' timestamps need no conversion
            all_data[accel_id].extend(rows)
            print(f"Loaded {len(rows):,} accel{accel_id} records from {filename} using format {chosen['name']} ({chosen['size']} bytes)")

    dfs = {}
    for accel_id in [1, 2, 3]:
        if all_data[accel_id]:
            # Ensure tuples -> DataFrame
            df = pd.DataFrame(all_data[accel_id], columns=['Timestamp', 'X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', origin='unix', errors='coerce')
            df = df.dropna(subset=['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            if not df.empty:
                start_time = df.index[0]
                df['Time_elapsed'] = (df.index - start_time).total_seconds()
            dfs[accel_id] = df if not df.empty else None
        else:
            print(f"Warning: No data found for accelerometer {accel_id}")
            dfs[accel_id] = None

    return dfs.get(1), dfs.get(2), dfs.get(3)


def load_and_process_spi_gyroscope_files(folder_path):
    """
    Load SPI gyro .bin files from gyro1.
    New single files: spi_gyro_YYYYMMDD_HHMMSS.bin -> <df> or <dd>
    Old chunks: {start}_chunk_####.bin -> <dd>
    """
    gyro_folder = os.path.join(folder_path, 'gyro1')
    if not os.path.exists(gyro_folder):
        print(f"Warning: SPI Gyroscope folder {gyro_folder} not found")
        return None

    bin_files = sorted([fn for fn in os.listdir(gyro_folder) if fn.lower().endswith('.bin')])
    if not bin_files:
        print("Warning: No .bin files found for SPI gyroscope")
        return None

    all_rows = []
    old_chunk_pat = re.compile(r'(\d+\.\d+)_chunk_\d+\.bin', re.IGNORECASE)

    for filename in bin_files:
        file_path = os.path.join(gyro_folder, filename)
        is_old = bool(old_chunk_pat.match(filename))
        is_new = filename.lower().startswith('spi_gyro_')

        if is_old:
            candidates = [
                {'name': 'dd', 'size': 16, 'fmt': '<dd', 'ts': ('d', 's')},
            ]
        else:
            candidates = [
                {'name': 'dd', 'size': 16, 'fmt': '<dd', 'ts': ('d', 's')},
                {'name': 'df', 'size': 12, 'fmt': '<df', 'ts': ('d', 's')},
            ]

        chosen = _detect_record_format(file_path, candidates)
        rows = _read_binary_records(file_path, chosen['fmt'])
        all_rows.extend(rows)
        print(f"Loaded {len(rows):,} SPI gyro records from {filename} using format {chosen['name']} ({chosen['size']} bytes)")

    if all_rows:
        df = pd.DataFrame(all_rows, columns=['Timestamp', 'Angular_Rate (deg/s)'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', origin='unix', errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)
        if not df.empty:
            start_time = df.index[0]
            df['Time_elapsed'] = (df.index - start_time).total_seconds()
        return df if not df.empty else None
    else:
        print("Warning: No data found for SPI gyroscope")
        return None


def load_and_process_i2c_gyroscope_files(folder_path):
    """
    Load I2C gyro .bin files from gyro_i2c1.
    New single files: i2c_gyro_YYYYMMDD_HHMMSS.bin -> <dffff> or <ddddd>
    Old chunks: {start}_chunk_####.bin -> <ddddd>
    """
    gyro_folder = os.path.join(folder_path, 'gyro_i2c1')
    if not os.path.exists(gyro_folder):
        print(f"Warning: I2C Gyroscope folder {gyro_folder} not found")
        return None

    bin_files = sorted([fn for fn in os.listdir(gyro_folder) if fn.lower().endswith('.bin')])
    if not bin_files:
        print("Warning: No .bin files found for I2C gyroscope")
        return None

    all_rows = []
    old_chunk_pat = re.compile(r'(\d+\.\d+)_chunk_\d+\.bin', re.IGNORECASE)

    for filename in bin_files:
        file_path = os.path.join(gyro_folder, filename)
        is_old = bool(old_chunk_pat.match(filename))
        is_new = filename.lower().startswith('i2c_gyro_')

        if is_old:
            candidates = [
                {'name': 'ddddd', 'size': 40, 'fmt': '<ddddd', 'ts': ('d', 's')},
            ]
        else:
            candidates = [
                {'name': 'ddddd', 'size': 40, 'fmt': '<ddddd', 'ts': ('d', 's')},
                {'name': 'dffff', 'size': 24, 'fmt': '<dffff', 'ts': ('d', 's')},
            ]

        chosen = _detect_record_format(file_path, candidates)
        rows = _read_binary_records(file_path, chosen['fmt'])
        all_rows.extend(rows)
        print(f"Loaded {len(rows):,} I2C gyro records from {filename} using format {chosen['name']} ({chosen['size']} bytes)")

    if all_rows:
        df = pd.DataFrame(all_rows, columns=['Timestamp', 'X (deg/s)', 'Y (deg/s)', 'Z (deg/s)', 'Temperature (°C)'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s', origin='unix', errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)
        if not df.empty:
            start_time = df.index[0]
            df['Time_elapsed'] = (df.index - start_time).total_seconds()
        return df if not df.empty else None
    else:
        print("Warning: No data found for I2C gyroscope")
        return None


def _read_binary_records(file_path, fmt):
    """
    Read the entire binary file as fixed-size records using struct format fmt.
    Returns a list of tuples (one per record).
    """
    # Ensure explicit endianness (default to little-endian)
    if not fmt.startswith(('<', '>', '!', '@', '=')):
        fmt = '<' + fmt
    rec_size = struct.calcsize(fmt)
    if rec_size <= 0:
        return []

    with open(file_path, 'rb') as f:
        data = f.read()

    # Trim any trailing partial record
    rem = len(data) % rec_size
    if rem:
        data = data[:-rem]

    return list(struct.iter_unpack(fmt, data))


def select_folder():
    """
    Provide an interactive prompt for the user to select a data folder.
    Returns:
    str: Path to the selected folder.
    """
    base_path = '/home/perseus/timmins_test/position_box'
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    print("Available data folders:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    print(f"{len(folders) + 1}. Enter custom path")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(folders):
                return os.path.join(base_path, folders[choice - 1])
            elif choice == len(folders) + 1:
                custom_path = input("Enter the custom path: ")
                if os.path.isdir(custom_path):
                    return custom_path
                else:
                    print("Invalid path. Please try again.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def analyze_complete_sampling_rate(accel_dfs, spi_gyro_df, i2c_gyro_df, save_dir="."):
    """
    Analyze and plot the sampling rate of all sensors.
    """
    import os as _os
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('Sampling Rate Analysis - All Position Sensors', fontsize=16)
    
    sensor_names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3', 'SPI Gyroscope', 'I2C Gyroscope']
    all_sensors = list(accel_dfs) + [spi_gyro_df, i2c_gyro_df]
    
    for i, (df, sensor_name) in enumerate(zip(all_sensors, sensor_names)):
        if df is not None:
            time_diff = df.index.to_series().diff().dt.total_seconds()
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                sampling_rate = 1.0 / time_diff
            sampling_rate = sampling_rate[np.isfinite(sampling_rate)]
            sampling_rate = sampling_rate[(sampling_rate > 0) & (sampling_rate < 1e6)]
            
            if i < 3:
                color, alpha = ['red', 'green', 'blue'][i], 0.7
            elif i == 3:
                color, alpha = 'magenta', 0.7
            else:
                color, alpha = 'cyan', 0.7
            
            axes[i].hist(sampling_rate, bins=100, edgecolor='black', color=color, alpha=alpha)
            axes[i].set_title(f'{sensor_name}')
            axes[i].set_xlabel('Sampling Rate (Hz)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlim(0, 2000)
            axes[i].grid(True, alpha=0.3)
            
            print(f"\nSampling Rate Statistics - {sensor_name}:")
            print(f"  Mean: {sampling_rate.mean():.2f} Hz")
            print(f"  Median: {sampling_rate.median():.2f} Hz")
            print(f"  Std Dev: {sampling_rate.std():.2f} Hz")
            print(f"  Min: {sampling_rate.min():.2f} Hz")
            print(f"  Max: {sampling_rate.max():.2f} Hz")
        else:
            axes[i].text(0.5, 0.5, 'No Data', horizontalalignment='center', 
                        verticalalignment='center', transform=axes[i].transAxes, fontsize=14)
            axes[i].set_title(f'{sensor_name}')
            axes[i].set_xlabel('Sampling Rate (Hz)')
            print(f"\n{sensor_name}: No data available")
    
    plt.tight_layout()
    plt.savefig(_os.path.join(save_dir, "plot1.png"))


def _shade_dropouts(ax, df, dropouts, color='y', alpha=0.15):
    if df is None or not dropouts:
        return
    t0 = df.index[0]
    for d in dropouts:
        try:
            start_s = (d['start'] - t0).total_seconds()
            end_s = (d['end'] - t0).total_seconds()
        except Exception:
            continue
        if np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s:
            ax.axvspan(start_s, end_s, color=color, alpha=alpha, lw=0)

def _percentile_bounds(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (lo, hi) percentiles from finite values of x. If not enough finite values, returns (None, None).
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return None, None
    lo, hi = np.percentile(x[finite], [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None, None
    return float(lo), float(hi)

def plot_complete_sensor_data(accel_dfs, spi_gyro_df, i2c_gyro_df, dropout_results=None, save_dir="."):
    """
    Plot complete sensor data with optional dropout shading.
    """
    fig, axes = plt.subplots(5, 3, figsize=(18, 20), sharex=True)
    fig.suptitle('Complete Position Sensor Data Analysis\n(3 Accelerometers + 1 SPI Gyroscope + 1 I2C Gyroscope)', fontsize=16)

    def _set_ylim_with_margin(ax, lo, hi, pad_frac=0.15, min_pad=0.2):
        if lo is None or hi is None:
            return
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return
        pad = max(span * pad_frac, min_pad)
        ax.set_ylim(lo - pad, hi + pad)

    accel_names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    accel_cols = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']

    # Rows 0-2: accelerometers (X/Y/Z in columns)
    for row, (name, df) in enumerate(zip(accel_names, accel_dfs)):
        for col, ax_name in enumerate(accel_cols):
            ax = axes[row, col]
            if df is None or ax_name not in df.columns:
                ax.text(0.5, 0.5, f'No {name} Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.grid(True, alpha=0.3)
                continue
            t = df['Time_elapsed'].to_numpy(dtype=float)
            y = df[ax_name].to_numpy(dtype=float)
            lo, hi = _percentile_bounds(y, 0.1, 99.9)
            ax.plot(t, y, lw=0.8)
            ax.set_title(f'{name} — {"XYZ"[col]} Axis')
            ax.set_ylabel(ax_name if col == 0 else '')
            ax.grid(True, alpha=0.3)
            if lo is not None:
                _set_ylim_with_margin(ax, lo, hi)
            if dropout_results and name in dropout_results:
                _shade_dropouts(ax, df, dropout_results[name].get('dropouts', []))

    # Row 3: SPI Gyro (Z only in first column)
    if spi_gyro_df is not None and 'Angular_Rate (deg/s)' in spi_gyro_df.columns:
        t_spi = spi_gyro_df['Time_elapsed'].to_numpy()
        z_spi = spi_gyro_df['Angular_Rate (deg/s)'].to_numpy()
        lo_spi, hi_spi = _percentile_bounds(z_spi, 0.1, 99.9)
        axes[3, 0].plot(t_spi, z_spi, 'm-', linewidth=0.8)
        axes[3, 0].set_ylabel('Angular Rate (deg/s)')
        axes[3, 0].set_title('SPI Gyroscope (ADXRS453) - Z Axis')
        axes[3, 0].grid(True, alpha=0.3)
        if lo_spi is not None:
            _set_ylim_with_margin(axes[3, 0], lo_spi, hi_spi)
        if dropout_results and 'SPI Gyroscope' in dropout_results:
            _shade_dropouts(axes[3, 0], spi_gyro_df, dropout_results['SPI Gyroscope'].get('dropouts', []))
    else:
        axes[3, 0].text(0.5, 0.5, 'No SPI Gyroscope Data', ha='center', va='center', transform=axes[3, 0].transAxes, fontsize=12)
        axes[3, 0].set_title('SPI Gyroscope (ADXRS453) - Z Axis')
        axes[3, 0].grid(True, alpha=0.3)

    axes[3, 1].set_visible(False)
    axes[3, 2].set_visible(False)

    # Row 4: I2C Gyro (X/Y/Z)
    if i2c_gyro_df is not None:
        t_i2c = i2c_gyro_df['Time_elapsed'].to_numpy()
        for col, comp in enumerate(['X (deg/s)', 'Y (deg/s)', 'Z (deg/s)']):
            ax = axes[4, col]
            if comp not in i2c_gyro_df.columns:
                ax.text(0.5, 0.5, 'No I2C Gyro Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.grid(True, alpha=0.3)
                continue
            y = i2c_gyro_df[comp].to_numpy()
            lo, hi = _percentile_bounds(y, 0.1, 99.9)
            ax.plot(t_i2c, y, linewidth=0.8)
            ax.set_title(f'I2C Gyroscope (IAM20380HT) - {"XYZ"[col]} Axis')
            ax.grid(True, alpha=0.3)
            if lo is not None:
                _set_ylim_with_margin(ax, lo, hi)
            if dropout_results and 'I2C Gyroscope' in dropout_results:
                _shade_dropouts(ax, i2c_gyro_df, dropout_results['I2C Gyroscope'].get('dropouts', []))
    else:
        for col in range(3):
            axes[4, col].text(0.5, 0.5, 'No I2C Gyroscope Data', ha='center', va='center', transform=axes[4, col].transAxes, fontsize=12)
            axes[4, col].set_title(f'I2C Gyroscope (IAM20380HT) - {"XYZ"[col]} Axis')
            axes[4, col].grid(True, alpha=0.3)

    # X labels only on bottom row
    for col in range(3):
        if axes[4, col].get_visible():
            axes[4, col].set_xlabel('Time (seconds)')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(os.path.join(save_dir, "plot.png"))


def plot_temperature_data(i2c_gyro_df, save_dir="."):
    """
    Plot temperature data from I2C gyroscope separately.
    """
    import os as _os
    if i2c_gyro_df is not None and 'Temperature (°C)' in i2c_gyro_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(i2c_gyro_df['Time_elapsed'].to_numpy(), i2c_gyro_df['Temperature (°C)'].to_numpy(), 'r-', linewidth=1)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('I2C Gyroscope Temperature Data')
        ax.grid(True, alpha=0.3)
        
        temp_mean = i2c_gyro_df['Temperature (°C)'].mean()
        temp_std = i2c_gyro_df['Temperature (°C)'].std()
        temp_range = i2c_gyro_df['Temperature (°C)'].max() - i2c_gyro_df['Temperature (°C)'].min()
        
        ax.text(0.02, 0.98, f'Mean: {temp_mean:.2f}°C\nStd: {temp_std:.3f}°C\nRange: {temp_range:.3f}°C', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(_os.path.join(save_dir, "plot2.png"))
        
        print(f"\nTemperature Statistics:")
        print(f"  Mean: {temp_mean:.3f}°C")
        print(f"  Std Dev: {temp_std:.3f}°C")
        print(f"  Min: {i2c_gyro_df['Temperature (°C)'].min():.3f}°C")
        print(f"  Max: {i2c_gyro_df['Temperature (°C)'].max():.3f}°C")
        print(f"  Range: {temp_range:.3f}°C")
    else:
        print("No temperature data available")


def _detect_dropouts_for_df(df, expected_rate_hz=None, threshold_factor=2.0):
    """
    Detect dropouts in a time-indexed DataFrame by inspecting gaps between consecutive timestamps.
    If expected_rate_hz is None, uses the median delta as the expected dt.
    Returns a dict with summary and a list of dropout records.
    """
    if df is None or len(df) < 2:
        return {
            'count': 0,
            'total_missing': 0,
            'worst_gap_s': 0.0,
            'expected_dt': None,
            'threshold_s': None,
            'dropouts': []
        }
    time_diff = df.index.to_series().diff().dt.total_seconds()
    diffs = time_diff.to_numpy()
    valid = diffs[np.isfinite(diffs)]
    if valid.size == 0:
        return {
            'count': 0,
            'total_missing': 0,
            'worst_gap_s': 0.0,
            'expected_dt': None,
            'threshold_s': None,
            'dropouts': []
        }
    median_dt = float(np.median(valid))
    expected_dt = (1.0 / float(expected_rate_hz)) if expected_rate_hz and expected_rate_hz > 0 else median_dt
    threshold_s = expected_dt * float(threshold_factor)

    dropout_idxs = np.where(diffs > threshold_s)[0]  # index i means gap between i-1 and i
    dropouts = []
    total_missing = 0
    worst_gap = 0.0

    for i in dropout_idxs:
        gap_s = float(diffs[i])
        start_ts = df.index[i - 1]
        end_ts = df.index[i]
        # estimate missing samples relative to expected dt
        est_missing = max(int(round(gap_s / expected_dt) - 1), 1)
        total_missing += est_missing
        worst_gap = max(worst_gap, gap_s)
        dropouts.append({
            'start': start_ts,
            'end': end_ts,
            'gap_seconds': gap_s,
            'estimated_missing_samples': est_missing
        })

    return {
        'count': int(len(dropout_idxs)),
        'total_missing': int(total_missing),
        'worst_gap_s': float(worst_gap),
        'expected_dt': float(expected_dt),
        'threshold_s': float(threshold_s),
        'dropouts': dropouts
    }


def _rolling_median_abs_deviation(s: pd.Series, window: int):
    med = s.rolling(window, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    return med, mad

def _highpass_1pole(x: np.ndarray, dt: float, fc: float) -> np.ndarray:
    """
    First-order high-pass (DC/slow drift removal). Stable, O(n).
    fc: cutoff Hz (e.g., 0.05–0.2 Hz). dt: sample period (s).
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(fc) or fc <= 0:
        return x - np.nanmean(x)
    rc = 1.0 / (2.0 * np.pi * fc)
    alpha = rc / (rc + dt)
    y = np.zeros_like(x)
    x0 = x[0] if x.size else 0.0
    for n in range(1, x.size):
        xn = x[n]
        xm1 = x[n - 1]
        y[n] = alpha * (y[n - 1] + (xn - xm1))
    # remove any tiny residual DC
    y -= np.nanmean(y)
    return y

def highpass_accel_df(df: pd.DataFrame, cols: Iterable[str], fc: float = 0.3) -> pd.DataFrame:
    """
    Apply 1-pole high-pass per axis using median dt from Time_elapsed or index.
    """
    if df is None or len(df) < 3:
        return df
    if 'Time_elapsed' in df.columns:
        t = df['Time_elapsed'].to_numpy(dtype=float)
        dt_med = float(np.median(np.diff(t)))
    else:
        dt = df.index.to_series().diff().dt.total_seconds().to_numpy()
        dt = dt[np.isfinite(dt) & (dt > 0)]
        dt_med = float(np.median(dt)) if dt.size else 0.0

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype(float).to_numpy()
        # simple NaN bridging
        if np.isnan(s).any():
            s = pd.Series(s).interpolate(limit_direction='both').to_numpy()
        out[c] = _highpass_1pole(s, dt=dt_med, fc=fc)
    return out

def despike_df(df: pd.DataFrame, columns, rate_hz: float, window_sec=0.05, n_sigmas=4.0, interpolate=True):
    """
    Hampel-type despiking per column:
    - window_sec: length of rolling window in seconds (short to catch spikes).
    - n_sigmas: threshold multiplier on MAD (scaled to ~sigma).
    - interpolate: if True, linearly interpolates removed spikes in time.
    Returns cleaned DataFrame and per-column replacement counts.
    """
    if df is None or rate_hz is None or rate_hz <= 0:
        return df, {}

    window = max(5, int(round(window_sec * rate_hz)) | 1)  # odd, >=5
    out = df.copy()
    report = {}

    for col in columns:
        if col not in out.columns:
            continue
        s = out[col].astype(float)
        med, mad = _rolling_median_abs_deviation(s, window)
        sigma_est = 1.4826 * mad
        thresh = n_sigmas * sigma_est + 1e-12
        mask = (s - med).abs() > thresh
        count = int(mask.sum())

        if count > 0:
            s_clean = s.mask(mask, np.nan)
            if interpolate and isinstance(out.index, pd.DatetimeIndex):
                s_clean = s_clean.interpolate(method='time', limit=window, limit_direction='both')
            else:
                s_clean = s_clean.where(~mask, med)
            out[col] = s_clean

        report[col] = {"replaced": count, "window": window}

    return out, report


def despike_all_sensors(accel_dfs, spi_gyro_df, i2c_gyro_df, expected_rates, window_sec=0.05, n_sigmas=4.0):
    """
    Apply despiking to all sensors. Returns cleaned (accel_dfs, spi_gyro_df, i2c_gyro_df)
    and writes a short report to outliers_report.txt.
    """
    lines = []
    cleaned_accels = []
    accel_names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    accel_cols = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']

    for name, df in zip(accel_names, accel_dfs):
        rate = (expected_rates or {}).get(name)
        cleaned, rep = despike_df(df, accel_cols, rate, window_sec, n_sigmas, interpolate=True)
        cleaned_accels.append(cleaned)
        if rep:
            total = sum(v["replaced"] for v in rep.values())
            lines.append(f"{name}: replaced {total} spikes across X/Y/Z (window={next(iter(rep.values()))['window']})")
        else:
            lines.append(f"{name}: no data")

    # SPI gyro
    spi_name = 'SPI Gyroscope'
    if spi_gyro_df is not None:
        spi_rate = (expected_rates or {}).get(spi_name)
        spi_cleaned, rep = despike_df(spi_gyro_df, ['Angular_Rate (deg/s)'], spi_rate, window_sec, n_sigmas, interpolate=True)
        spi_gyro_df = spi_cleaned
        lines.append(f"{spi_name}: replaced {rep.get('Angular_Rate (deg/s)', {}).get('replaced', 0)} spikes")
    else:
        lines.append(f"{spi_name}: no data")

    # I2C gyro (do not touch Temperature)
    i2c_name = 'I2C Gyroscope'
    if i2c_gyro_df is not None:
        i2c_rate = (expected_rates or {}).get(i2c_name)
        i2c_cleaned, rep = despike_df(i2c_gyro_df, ['X (deg/s)', 'Y (deg/s)', 'Z (deg/s)'], i2c_rate, window_sec, n_sigmas, interpolate=True)
        i2c_gyro_df = i2c_cleaned
        total = sum(v["replaced"] for v in rep.values())
        lines.append(f"{i2c_name}: replaced {total} spikes across X/Y/Z")
    else:
        lines.append(f"{i2c_name}: no data")

    try:
        with open("outliers_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print("Outlier report saved to: outliers_report.txt")
    except Exception as e:
        print(f"Could not save outlier report: {e}")

    return (tuple(cleaned_accels), spi_gyro_df, i2c_gyro_df)



def analyze_dropouts(accel_dfs, spi_gyro_df, i2c_gyro_df, threshold_factor=2.0, expected_rates=None, report_path="dropouts_report.txt"):
    """
    Analyze dropouts for all sensors and return per-sensor results.
    """
    sensors = [
        ('Accelerometer 1', accel_dfs[0]),
        ('Accelerometer 2', accel_dfs[1]),
        ('Accelerometer 3', accel_dfs[2]),
        ('SPI Gyroscope', spi_gyro_df),
        ('I2C Gyroscope', i2c_gyro_df),
    ]
    expected_rates = expected_rates or {}

    lines = []
    lines.append("=" * 80)
    lines.append(f"DROPOUT ANALYSIS (threshold_factor={threshold_factor})")
    lines.append("=" * 80)

    print("\n" + "=" * 80)
    print(f"DROPOUT ANALYSIS (threshold_factor={threshold_factor})")
    print("=" * 80)

    results = {}
    for name, df in sensors:
        rate = expected_rates.get(name)
        res = _detect_dropouts_for_df(df, expected_rate_hz=rate, threshold_factor=threshold_factor)
        results[name] = res

        if df is None or len(df) < 2:
            msg = f"{name}: No data"
            print(msg)
            lines.append(msg)
            continue

        msg = (f"{name}: dropouts={res['count']}, total_est_missing={res['total_missing']}, "
               f"worst_gap={res['worst_gap_s']:.6f}s, expected_dt={res['expected_dt']:.6f}s, "
               f"threshold={res['threshold_s']:.6f}s")
        print(msg)
        lines.append(msg)

        for d in res['dropouts'][:10]:
            detail = (f"  gap {d['gap_seconds']:.6f}s from {d['start']} -> {d['end']} "
                      f"(~{d['estimated_missing_samples']} missing)")
            lines.append(detail)
        if res['count'] > 10:
            lines.append(f"  ... {res['count'] - 10} more")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nDropout report saved to: {report_path}")
    except Exception as e:
        print(f"Could not save dropout report: {e}")

    return results

def average_data(accel_dfs, spi_gyro_df, i2c_gyro_df):
    sensors = [
        ('Accelerometer 1', accel_dfs[0]),
        ('Accelerometer 2', accel_dfs[1]),
        ('Accelerometer 3', accel_dfs[2]),
        ('SPI Gyroscope', spi_gyro_df),
        ('I2C Gyroscope', i2c_gyro_df),
    ]

    sensors[0] = np.mean(accel_dfs[0], axis=0)
    sensors[1] = np.mean(accel_dfs[1], axis=0)
    sensors[2] = np.mean(accel_dfs[2], axis=0)
    sensors[3] = np.mean(spi_gyro_df, axis=0)
    sensors[4] = np.mean(i2c_gyro_df, axis=0)

    return sensors

def analyze_1s_displacement_drift(
    t: ArrayLike,
    disp_df: pd.DataFrame,
    cols: Iterable[str],
    threshold_mm: float = 1.0,
    window_sec: float = 1.0,
    tol_frac: float = 0.05,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    For each ~1-second span, compute displacement change s(t+1s) - s(t).
    Filters out windows whose actual duration is not within window_sec±tol_frac.
    Returns a DataFrame (mm) indexed by start time in seconds.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) != len(disp_df):
        return pd.DataFrame()
    # Ensure monotonic time
    if not np.all(np.diff(t) > 0):
        order = np.argsort(t)
        t = t[order]
        disp_df = disp_df.iloc[order]

    idx = np.arange(len(t))
    end_idx = np.searchsorted(t, t + window_sec, side="left")
    valid = end_idx < len(t)
    start_idx = idx[valid]
    end_idx = end_idx[valid]

    # Keep only windows with actual span close to requested
    dt_win = t[end_idx] - t[start_idx]
    lo = window_sec * (1.0 - tol_frac)
    hi = window_sec * (1.0 + tol_frac)
    
    # Fix: Use bitwise & instead of logical and for arrays
    mask = (dt_win >= lo) & (dt_win <= hi)

    start_idx = start_idx[mask]
    end_idx = end_idx[mask]

    if start_idx.size == 0:
        if verbose:
            print(f"Drift: no valid windows within {window_sec:.2f}s ±{tol_frac*100:.1f}% (kept 0).")
        return pd.DataFrame()

    S = disp_df[list(cols)].to_numpy(dtype=float)  # meters
    delta = S[end_idx, :] - S[start_idx, :]        # meters over ~1 s
    drift_df = pd.DataFrame(
        delta * 1000.0,  # mm
        index=pd.Index(t[start_idx], name="t_start_s"),
        columns=list(cols),
    )
    if verbose:
        kept = start_idx.size
        total = np.count_nonzero(valid)
        print(f"Drift: kept {kept:,}/{total:,} windows within [{lo:.3f}, {hi:.3f}] s "
              f"(dt_win median={np.median(dt_win[mask]):.3f}s, 95th={np.percentile(dt_win[mask],95):.3f}s).")
    return drift_df

def analyze_1s_displacement_drift_windowed(
    t: ArrayLike,
    vel_df: pd.DataFrame,
    disp_df: pd.DataFrame,
    cols: Iterable[str],
    window_sec: float = 1.0,
    tol_frac: float = 0.05,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute windowed 2x-integration displacement per axis using:
      drift = (S[b] - S[a]) - V[a] * (t[b] - t[a]),
    which removes dependence on pre-window velocity bias.
    Returns a DataFrame (mm) indexed by start time in seconds.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or (len(t) != len(vel_df) or len(t) != len(disp_df)):
        return pd.DataFrame()

    # Ensure monotonic time
    if not np.all(np.diff(t) > 0):
        order = np.argsort(t)
        t = t[order]
        vel_df = vel_df.iloc[order]
        disp_df = disp_df.iloc[order]

    idx = np.arange(len(t))
    end_idx = np.searchsorted(t, t + window_sec, side="left")
    valid = end_idx < len(t)
    start_idx = idx[valid]
    end_idx = end_idx[valid]

    dt_win = t[end_idx] - t[start_idx]
    lo = window_sec * (1.0 - tol_frac)
    hi = window_sec * (1.0 + tol_frac)
    mask = (dt_win >= lo) & (dt_win <= hi)

    if not np.any(mask):
        if verbose:
            print(f"Drift: no valid windows within {window_sec:.2f}s ±{tol_frac*100:.1f}% (kept 0).")
        return pd.DataFrame()

    start_idx = start_idx[mask]
    end_idx = end_idx[mask]
    dt_win = dt_win[mask]

    V = vel_df[list(cols)].to_numpy(dtype=float)   # m/s
    S = disp_df[list(cols)].to_numpy(dtype=float)  # m
    # windowed double integral: (S[b]-S[a]) - V[a]*(Δt)
    delta = S[end_idx, :] - S[start_idx, :] - V[start_idx, :] * dt_win[:, None]  # meters
    drift_df = pd.DataFrame(
        delta * 1000.0,  # mm
        index=pd.Index(t[start_idx], name="t_start_s"),
        columns=list(cols),
    )
    if verbose:
        kept = start_idx.size
        total = np.count_nonzero(valid)
        print(f"Drift: kept {kept:,}/{total:,} windows within [{lo:.3f}, {hi:.3f}] s "
              f"(dt_win median={np.median(dt_win):.3f}s, 95th={np.percentile(dt_win,95):.3f}s).")
    return drift_df


def evaluate_submillimeter_one_second(accel_dfs, threshold_mm=1.0, save_dir=".", hp_fc: float = 0.3):
    """
    For each accelerometer, high-pass acceleration to remove slow bias,
    integrate twice to velocity/disp (m/s, m), compute 1-second windowed drift (mm).
    """
    os.makedirs(save_dir, exist_ok=True)
    accel_names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    accel_cols = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']

    for name, df in zip(accel_names, accel_dfs):
        if df is None or len(df) < 3:
            print(f"{name}: No data for 1-second drift evaluation")
            continue

        t = df['Time_elapsed'].to_numpy(dtype=float)

        # High-pass filter to suppress slow bias/gravity leakage
        hp_df = highpass_accel_df(df, accel_cols, fc=hp_fc)

        # Double integration to velocity and displacement (m/s, m)
        _, _, vel_df, disp_df = mean_center_and_double_integrate(hp_df, t, accel_cols, interpolate_nans=True)

        # Windowed 2x integration over true 1 s spans
        drift_mm_df = analyze_1s_displacement_drift_windowed(
            t, vel_df, disp_df, accel_cols, window_sec=1.0, tol_frac=0.05, verbose=True
        )
        if drift_mm_df.empty:
            print(f"{name}: Not enough span for 1-second drift windows")
            continue

        mag_mm = np.sqrt((drift_mm_df.values ** 2).sum(axis=1))
        mag_mm_series = pd.Series(mag_mm, index=drift_mm_df.index, name="drift_mm")

        n = len(mag_mm_series)
        violations = int((mag_mm_series > threshold_mm).sum())
        pct_viol = 100.0 * violations / n
        mean_mag = float(np.nanmean(mag_mm_series))
        p95 = float(np.nanpercentile(mag_mm_series, 95))
        worst = float(np.nanmax(mag_mm_series))

        print(f"\n{name} — 1 s displacement drift (windowed, HP fc={hp_fc:.3f} Hz):")
        print(f"  Windows: {n:,}")
        print(f"  Mean: {mean_mag:.3f} mm, 95th: {p95:.3f} mm, Worst: {worst:.3f} mm")
        print(f"  Threshold: {threshold_mm:.3f} mm — Violations: {violations:,} ({pct_viol:.2f}%)")

        for ax in accel_cols:
            ax_abs = drift_mm_df[ax].abs()
            print(f"  {ax}: mean={ax_abs.mean():.3f} mm, 95th={np.percentile(ax_abs,95):.3f} mm, worst={ax_abs.max():.3f} mm")

        x99 = float(np.nanpercentile(mag_mm_series, 99))
        xmax = max(threshold_mm * 3.0, x99)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.hist(mag_mm_series, bins=100, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.axvline(threshold_mm, color='red', linestyle='--', label=f'{threshold_mm:.1f} mm')
        ax1.set_title(f'{name} — 1 s Drift Magnitude Histogram')
        ax1.set_xlabel('Drift (mm)')
        ax1.set_ylabel('Count')
        ax1.set_xlim(0, xmax)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(mag_mm_series.index.values, mag_mm_series.values, color='steelblue', lw=0.8)
        ax2.axhline(threshold_mm, color='red', linestyle='--', label=f'{threshold_mm:.1f} mm')
        ax2.set_title(f'{name} — 1 s Drift Magnitude vs Time')
        ax2.set_xlabel('t_start (s)')
        ax2.set_ylabel('Drift (mm)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        out_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}_1s_drift.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved plot: {out_path}")

def evaluate_long_term_drift(accel_dfs, save_dir=".", hp_fc: float = 0.2, window_sec: Optional[float] = 60.0):
    """
    Bias-robust long-term drift:
      - high-pass accel at fc to suppress DC/slow bias,
      - integrate to velocity/displacement (m/s, m),
      - summarize displacement stats,
      - optionally compute sliding-window (window_sec) drift using windowed 2× integration.
    """
    os.makedirs(save_dir, exist_ok=True)
    accel_names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    accel_cols = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']

    for name, df in zip(accel_names, accel_dfs):
        if df is None or len(df) < 3:
            print(f"{name}: No data for long-term drift evaluation")
            continue

        t = df['Time_elapsed'].to_numpy(dtype=float)

        # High-pass to remove slow bias/gravity leakage
        hp_df = highpass_accel_df(df, accel_cols, fc=hp_fc)

        # Integrate once and twice
        _, _, vel_hp, disp_hp = mean_center_and_double_integrate(hp_df, t, accel_cols, interpolate_nans=True)

        print(f"\n{name} — Long-term drift (HP fc={hp_fc:.3f} Hz):")
        for ax in accel_cols:
            s_m = disp_hp[ax].to_numpy(dtype=float)
            if np.isfinite(s_m).any():
                span_mm = float((np.nanmax(s_m) - np.nanmin(s_m)) * 1000.0)
                rms_mm = float(np.sqrt(np.nanmean((s_m - np.nanmean(s_m))**2)) * 1000.0)
            else:
                span_mm = rms_mm = float('nan')
            print(f"  {ax}: span={span_mm:.2f} mm, RMS={rms_mm:.2f} mm")

        # Plot HP displacement
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        colors = ['r', 'g', 'b']
        for i, ax_name in enumerate(accel_cols):
            s_m = disp_hp[ax_name].to_numpy(dtype=float)
            axs[i].plot(t, s_m * 1000.0, color=colors[i], lw=0.8, label=f'{ax_name} disp (HP)')
            axs[i].set_ylabel('mm')
            axs[i].grid(True, alpha=0.3)
            axs[i].legend()
        axs[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{name} — Displacement (HP @ {hp_fc:.3f} Hz)')
        out_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}_displacement_trend.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved plot: {out_path}")

        # Optional: sliding-window drift (keeps metric local; won’t blow up)
        if window_sec and window_sec > 0:
            drift_mm_df = analyze_1s_displacement_drift_windowed(
                t, vel_hp, disp_hp, accel_cols, window_sec=float(window_sec), tol_frac=0.05, verbose=False
            )
            if not drift_mm_df.empty:
                mag_mm = np.sqrt((drift_mm_df.values ** 2).sum(axis=1))
                mean_mag = float(np.nanmean(mag_mm))
                p95 = float(np.nanpercentile(mag_mm, 95))
                worst = float(np.nanmax(mag_mm))
                print(f"  Sliding-window drift ({window_sec:.0f}s): mean={mean_mag:.3f} mm, 95th={p95:.3f} mm, worst={worst:.3f} mm")

def _estimate_rate_hz_from_index(idx: pd.DatetimeIndex) -> float:
    if idx is None or len(idx) < 2:
        return float('nan')
    diffs = idx.to_series().diff().dt.total_seconds().to_numpy()
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float('nan')
    med_dt = float(np.median(diffs))
    return 1.0 / med_dt if med_dt > 0 else float('nan')


def _rolling_robust_sigma(s: pd.Series, win: int) -> pd.Series:
    # Robust sigma ~ 1.4826 * MAD around rolling median
    med = s.rolling(win, center=True, min_periods=max(3, win//5)).median()
    dev = (s - med).abs()
    mad = dev.rolling(win, center=True, min_periods=max(3, win//5)).median()
    return 1.4826 * mad

def _mask_to_segments(idx: pd.DatetimeIndex, mask: np.ndarray, min_segment_sec: float, pad_sec: float = 0.2):
    """
    Convert a boolean mask over a DatetimeIndex into merged [start,end] segments,
    requiring each segment to last at least min_segment_sec. Pads each end by pad_sec.
    """
    segments = []
    if idx is None or len(idx) == 0 or mask is None or mask.size != len(idx):
        return segments
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return segments

    # Find runs of True
    start_i = None
    for i, m in enumerate(mask):
        if m and start_i is None:
            start_i = i
        if (not m or i == len(mask) - 1) and start_i is not None:
            end_i = i if not m else i  # inclusive
            t0 = idx[start_i]
            t1 = idx[end_i]
            dur = (t1 - t0).total_seconds()
            if dur >= min_segment_sec:
                # pad
                seg_start = t0 - pd.Timedelta(seconds=pad_sec)
                seg_end = t1 + pd.Timedelta(seconds=pad_sec)
                segments.append((seg_start, seg_end))
            start_i = None

    # Merge overlapping/adjacent segments
    if not segments:
        return segments
    segments.sort(key=lambda x: x[0])
    merged = [segments[0]]
    for s, e in segments[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

def find_calm_segments_for_accel(df: pd.DataFrame,
                                 cols=('X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'),
                                 window_sec: float = 0.5,
                                 sigma_thresh: float = 0.03,
                                 min_segment_sec: float = 5.0,
                                 pad_sec: float = 0.2):
    """
    Identify calm segments where rolling robust sigma (per axis) is below sigma_thresh.
    - window_sec: rolling window length in seconds
    - sigma_thresh: m/s^2 (e.g., 0.03 ≈ 3 mg) threshold for robust sigma
    - min_segment_sec: keep segments at least this long
    - pad_sec: pad each segment edge to be conservative
    Returns list of (start_ts, end_ts).
    """
    if df is None or len(df) < 3:
        return []

    rate_hz = _estimate_rate_hz_from_index(df.index)
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        return []
    win = max(5, int(round(window_sec * rate_hz)) | 1)  # odd window, >=5

    masks = []
    for c in cols:
        if c not in df.columns:
            continue
        sig = _rolling_robust_sigma(df[c].astype(float), win)
        masks.append((sig <= sigma_thresh).to_numpy())
    if not masks:
        return []

    # Calm only if all present axes are calm
    calm_mask = np.logical_and.reduce(masks)
    return _mask_to_segments(df.index, calm_mask, min_segment_sec=min_segment_sec, pad_sec=pad_sec)

def _intersect_two_segment_lists(a, b):
    """
    Intersect two lists of (start_ts, end_ts) intervals (Datetime).
    Returns a new list with overlaps only.
    """
    out = []
    i = j = 0
    a = sorted(a, key=lambda x: x[0])
    b = sorted(b, key=lambda x: x[0])
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if s < e:
            out.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return out

def find_common_calm_segments(accel_dfs, window_sec=0.5, sigma_thresh=0.03, min_segment_sec=5.0, pad_sec=0.2):
    """
    Find calm segments common to all available accelerometers.
    """
    names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    cols = ('X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)')
    per_accel = []
    for name, df in zip(names, accel_dfs):
        segs = find_calm_segments_for_accel(df, cols=cols, window_sec=window_sec,
                                            sigma_thresh=sigma_thresh,
                                            min_segment_sec=min_segment_sec,
                                            pad_sec=pad_sec)
        if segs:
            print(f"{name}: found {len(segs)} calm segments")
            per_accel.append(segs)
        else:
            print(f"{name}: no calm segments")
    if not per_accel:
        return []

    # Intersect across all lists
    common = per_accel[0]
    for segs in per_accel[1:]:
        common = _intersect_two_segment_lists(common, segs)
        if not common:
            break
    # Drop very short segments if any
    common = [(s, e) for (s, e) in common if (e - s).total_seconds() >= min_segment_sec]
    total_sec = sum((e - s).total_seconds() for s, e in common)
    print(f"Common calm segments: {len(common)} (total {total_sec:.2f} s)")
    return common

def evaluate_submillimeter_one_second_calm(accel_dfs, segments, threshold_mm=1.0, save_dir=".", hp_fc: float = 0.3):
    """
    Calm-only evaluation with high-pass filtered acceleration before 2× integration.
    Uses windowed drift to remove pre-window velocity term.
    """
    if not segments:
        print("No calm segments provided; skipping calm-only evaluation.")
        return

    os.makedirs(save_dir, exist_ok=True)
    accel_names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    accel_cols = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']

    for name, df in zip(accel_names, accel_dfs):
        if df is None or len(df) < 3:
            print(f"{name}: No data for calm-only evaluation")
            continue

        all_drift_mags = []
        tstarts = []
        used_secs = 0.0

        for (s_ts, e_ts) in segments:
            df_seg = df.loc[s_ts:e_ts]
            if df_seg is None or len(df_seg) < 3:
                continue
            t = df_seg['Time_elapsed'].to_numpy(dtype=float)
            if (t[-1] - t[0]) < 1.0:
                continue

            # High-pass per segment
            hp_seg = highpass_accel_df(df_seg, accel_cols, fc=hp_fc)

            # Integrate to V and S
            _, _, vel_df, disp_df = mean_center_and_double_integrate(hp_seg, t, accel_cols, interpolate_nans=True)

            drift_mm_df = analyze_1s_displacement_drift_windowed(
                t, vel_df, disp_df, accel_cols, window_sec=1.0, tol_frac=0.05, verbose=False
            )
            if drift_mm_df.empty:
                continue

            mag_mm = np.sqrt((drift_mm_df.values ** 2).sum(axis=1))
            all_drift_mags.append(mag_mm)
            tstarts.append(drift_mm_df.index.values)
            used_secs += (df_seg.index[-1] - df_seg.index[0]).total_seconds()

        if not all_drift_mags:
            print(f"{name}: No usable calm windows >= 1 s")
            continue

        mag_mm_all = np.concatenate(all_drift_mags)
        tstart_all = np.concatenate(tstarts)

        n = len(mag_mm_all)
        violations = int((mag_mm_all > threshold_mm).sum())
        pct_viol = 100.0 * violations / n
        mean_mag = float(np.nanmean(mag_mm_all))
        p95 = float(np.nanpercentile(mag_mm_all, 95))
        worst = float(np.nanmax(mag_mm_all))

        print(f"\n{name} — Calm-only 1 s displacement drift (windowed, HP fc={hp_fc:.3f} Hz):")
        print(f"  Calm coverage used: {used_secs:.2f} s across {len(segments)} segments")
        print(f"  Windows: {n:,}")
        print(f"  Mean: {mean_mag:.3f} mm, 95th: {p95:.3f} mm, Worst: {worst:.3f} mm")
        print(f"  Threshold: {threshold_mm:.3f} mm — Violations: {violations:,} ({pct_viol:.2f}%)")
        print(f"  Result: {'PASS' if violations == 0 else f'FAIL — {violations} windows > {threshold_mm:.1f} mm (worst {worst:.3f} mm)'}")

        # Plots
        x99 = float(np.nanpercentile(mag_mm_all, 99))
        xmax = max(threshold_mm * 3.0, x99)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.hist(mag_mm_all, bins=100, color='seagreen', edgecolor='black', alpha=0.8)
        ax1.axvline(threshold_mm, color='red', linestyle='--', label=f'{threshold_mm:.1f} mm')
        ax1.set_title(f'{name} — Calm-only 1 s Drift Histogram')
        ax1.set_xlabel('Drift (mm)')
        ax1.set_ylabel('Count')
        ax1.set_xlim(0, xmax)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(tstart_all, mag_mm_all, color='seagreen', lw=0.8)
        ax2.axhline(threshold_mm, color='red', linestyle='--', label=f'{threshold_mm:.1f} mm')
        ax2.set_title(f'{name} — Calm-only 1 s Drift vs Time (sensor seconds)')
        ax2.set_xlabel('t_start (s)')
        ax2.set_ylabel('Drift (mm)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        out_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}_1s_drift_calm.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved plot: {out_path}")

def _print_timebase_check(name: str, df: pd.DataFrame):
    if df is None or len(df) < 3:
        print(f"{name}: no data for timebase check")
        return
    dt = df.index.to_series().diff().dt.total_seconds().to_numpy()
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        print(f"{name}: no positive dt")
        return
    print(f"{name} timebase: median dt={np.median(dt):.6f}s, 1%={np.percentile(dt,1):.6f}s, 99%={np.percentile(dt,99):.6f}s, "
          f"max gap={dt.max():.3f}s")

def _print_accel_scale_sanity(accel_dfs):
    names = ['Accelerometer 1', 'Accelerometer 2', 'Accelerometer 3']
    cols = ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)']
    for name, df in zip(names, accel_dfs):
        if df is None or len(df) < 10:
            print(f"{name}: no data for scale sanity")
            continue
        mg = 1000.0 / 9.80665
        stats = []
        for c in cols:
            if c in df.columns:
                s = df[c].astype(float)
                med = float(s.median())
                mad = float((s - med).abs().median())
                sigma = 1.4826 * mad
                stats.append(f"{c}: median={med:.4f} m/s², robust σ={sigma*mg:.2f} mg")
        print(f"{name} accel scale sanity -> " + "; ".join(stats))

def center_data():
    folder_path = select_folder()
    print(f"Selected folder: {folder_path}")
    
    # Load all sensor data
    print("\nLoading sensor data...")
    print("Loading accelerometer data...")
    accel_df1, accel_df2, accel_df3 = load_and_process_accelerometer_files(folder_path)
    accel_dfs = (accel_df1, accel_df2, accel_df3)
    
    print("Loading SPI gyroscope data...")
    spi_gyro_df = load_and_process_spi_gyroscope_files(folder_path)
    
    print("Loading I2C gyroscope data...")
    i2c_gyro_df = load_and_process_i2c_gyroscope_files(folder_path)

    # Expected ODRs (firmware-configured)
    expected_rates = {
        'Accelerometer 1': 1000,
        'Accelerometer 2': 1000,
        'Accelerometer 3': 1000,
        'SPI Gyroscope': 1000,
        'I2C Gyroscope': 1000,
    }
    
    # Despike signals before plotting/analysis (50 ms window, ~4σ threshold)
    print("\nDespiking short spikes...")
    accel_dfs, spi_gyro_df, i2c_gyro_df = despike_all_sensors(
        accel_dfs, spi_gyro_df, i2c_gyro_df, expected_rates, window_sec=0.05, n_sigmas=4.0
    )

    print("\nTimebase checks:")
    for i, df in enumerate(accel_dfs, 1):
        _print_timebase_check(f"Accelerometer {i}", df)
    
    _print_timebase_check("SPI Gyroscope", spi_gyro_df)
    _print_timebase_check("I2C Gyroscope", i2c_gyro_df)

    print("\nAcceleration scale sanity (robust σ in mg):")
    _print_accel_scale_sanity(accel_dfs)

    # Make a figures directory within the selected dataset
    figures_dir = os.path.join(folder_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Check for dropouts (needed for shading)
    print("\nChecking for dropouts...")
    dropout_results = analyze_dropouts(
        accel_dfs, spi_gyro_df, i2c_gyro_df, threshold_factor=2.0, expected_rates=expected_rates,
        report_path=os.path.join(folder_path, "dropouts_report.txt")
    )
    
    # Plot all sensor data with dropout shading
    print("\nGenerating plots...")
    plot_complete_sensor_data(accel_dfs, spi_gyro_df, i2c_gyro_df, dropout_results=dropout_results, save_dir=figures_dir)
    
    # Analyze sampling rates
    analyze_complete_sampling_rate(accel_dfs, spi_gyro_df, i2c_gyro_df, save_dir=figures_dir)
    
    # Plot temperature data separately
    plot_temperature_data(i2c_gyro_df, save_dir=figures_dir)

    # Evaluate 1 mm per 1 s displacement drift (accelerometers)
    print("\nEvaluating 1 mm per 1 s displacement drift (accelerometers)...")
    #evaluate_submillimeter_one_second(accel_dfs, threshold_mm=1.0, save_dir=figures_dir, hp_fc=0.3)

    # Find calm segments common to all accelerometers and evaluate within them
    print("\nFinding calm segments and evaluating calm-only drift...")
    #calm_segments = find_common_calm_segments(
     #   accel_dfs,
      #  window_sec=0.5,
       # sigma_thresh=0.03,
        #min_segment_sec=5.0,
       # pad_sec=0.2
    #)
 #   if calm_segments:
        #evaluate_submillimeter_one_second_calm(accel_dfs, calm_segments, threshold_mm=1.0, save_dir=figures_dir, hp_fc=0.3)
  #  else:
   #     print("No common calm segments detected; skipping calm-only evaluation.")

    # Evaluate long-term drift (accelerometers)
    print("\nEvaluating long-term drift (accelerometers)...")
   # evaluate_long_term_drift(accel_dfs, save_dir=figures_dir)

    # Show plots in a window
    plt.show()

    # Print comprehensive data summary
    print("\n" + "="*80)
    print("COMPLETE POSITION SENSOR DATASET SUMMARY")
    print("="*80)
    
    for i, df in enumerate(accel_dfs, 1):
        if df is not None:
            print(f"\nAccelerometer {i} (ADXL355):")
            print(f"  Total samples: {len(df):,}")
            print(f"  Time span: {df.index[-1] - df.index[0]}")
            print(f"  First timestamp: {df.index[0]}")
            print(f"  Last timestamp: {df.index[-1]}")
            print(f"  Average data rate: {len(df) / max(df['Time_elapsed'].iloc[-1], 1e-9):.2f} Hz")
            print(f"  X-axis range: {df['X (m/s^2)'].min():.3f} to {df['X (m/s^2)'].max():.3f} m/s²")
            print(f"  Y-axis range: {df['Y (m/s^2)'].min():.3f} to {df['Y (m/s^2)'].max():.3f} m/s²")
            print(f"  Z-axis range: {df['Z (m/s^2)'].min():.3f} to {df['Z (m/s^2)'].max():.3f} m/s²")
        else:
            print(f"\nAccelerometer {i}: No data available")
    
    if spi_gyro_df is not None:
        print(f"\nSPI Gyroscope (ADXRS453):")
        print(f"  Total samples: {len(spi_gyro_df):,}")
        print(f"  Time span: {spi_gyro_df.index[-1] - spi_gyro_df.index[0]}")
        print(f"  First timestamp: {spi_gyro_df.index[0]}")
        print(f"  Last timestamp: {spi_gyro_df.index[-1]}")
        print(f"  Average data rate: {len(spi_gyro_df) / max(spi_gyro_df['Time_elapsed'].iloc[-1], 1e-9):.2f} Hz")
        print(f"  Angular rate range: {spi_gyro_df['Angular_Rate (deg/s)'].min():.3f} to {spi_gyro_df['Angular_Rate (deg/s)'].max():.3f} deg/s")
    else:
        print(f"\nSPI Gyroscope: No data available")
    
    if i2c_gyro_df is not None:
        print(f"\nI2C Gyroscope (IAM20380HT):")
        print(f"  Total samples: {len(i2c_gyro_df):,}")
        print(f"  Time span: {i2c_gyro_df.index[-1] - i2c_gyro_df.index[0]}")
        print(f"  First timestamp: {i2c_gyro_df.index[0]}")
        print(f"  Last timestamp: {i2c_gyro_df.index[-1]}")
        print(f"  Average data rate: {len(i2c_gyro_df) / max(i2c_gyro_df['Time_elapsed'].iloc[-1], 1e-9):.2f} Hz")
        print(f"  X-axis range: {i2c_gyro_df['X (deg/s)'].min():.3f} to {i2c_gyro_df['X (deg/s)'].max():.3f} deg/s")
        print(f"  Y-axis range: {i2c_gyro_df['Y (deg/s)'].min():.3f} to {i2c_gyro_df['Y (deg/s)'].max():.3f} deg/s")
        print(f"  Z-axis range: {i2c_gyro_df['Z (deg/s)'].min():.3f} to {i2c_gyro_df['Z (deg/s)'].max():.3f} deg/s")
        print(f"  Temperature range: {i2c_gyro_df['Temperature (°C)'].min():.2f} to {i2c_gyro_df['Temperature (°C)'].max():.2f} °C")
    else:
        print(f"\nI2C Gyroscope: No data available")

def main():
    center_data()

if __name__ == "__main__":
    main()
