# %%
import torch
from torch.utils.data import Dataset, IterableDataset
import polars as pl
import pandas as pd 
import pyarrow.parquet as pq
import pyarrow.dataset as pds
from tqdm import tqdm
import os
import random
from collections import deque
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import median_filter
import math 

# %% default_parameters
DEFAULT_CHUNK_DURATION = 4.0 #seconds
DEFAULT_FORECAST_DURATION = 0.0 #seconds
DEFAULT_SPATIAL_LENGTH = 0.25 #meters
DEFAULT_M = 64 # spatial sampling


# %% basic dataset
class WaveformDataset(Dataset):
    def __init__(self, data_tensor, dt=0.01):
        self.data = torch.tensor(data_tensor, dtype=torch.float32)
        self.N_patients, self.N_windows, self.T, self.M = self.data.shape
        self.dt = dt

    def __len__(self):
        return self.N_patients * self.N_windows

    def __getitem__(self, idx):
        p = idx // self.N_windows
        w = idx % self.N_windows
        seg = self.data[p, w]  # [T, M]
        x = seg[:, :2]         # ECG + PPG
        bp = seg[:, 2]         # BP (ABP or CVP)
        map_val = seg[:, 3].mean()
        T = seg.shape[0]
        t = torch.linspace(0, (T - 1) * self.dt, T).unsqueeze(1)
        t = (t - t.mean()) / t.std()

        return {
            "input": x,         # [T, 2]
            "output": bp,       # [T]
            "map": map_val, # scalar
            "t": t          # [T, 1]
        }

def build_coords(
    dt: float,
    T: int,
    M: int,
    L: float = 0.25,
    normalize: bool = True
):
    """
    Builds normalized t, t_input, x_locs, and coords grid for DeepONet/PINN input.

    Args:
        dt (float): Time step (1 / fs)
        T (int): Total number of time steps
        M (int): Number of spatial grid points
        L (float): Vessel length in meters (default = 0.25m)

    Returns:
        t (Tensor): [T, 1] normalized time vector
        t_input (Tensor): [T_input, 1] time for input window
        x_locs (Tensor): [M, 1] spatial grid
        coords (Tensor): [T, M, 2] full space-time coordinate grid
    """
    # Time axis
    t = torch.arange(T, dtype=torch.float32).unsqueeze(1) * dt

    # Space axis
    x_locs = torch.linspace(0.0, L, M).unsqueeze(1)  # [M, 1]
    x_grid = x_locs.repeat(1, T).T.unsqueeze(-1)     # [T, M, 1]
    t_grid = t.repeat(1, M).unsqueeze(-1)            # [T, M, 1]
    coords = torch.cat([x_grid, t_grid], dim=-1)     # [T, M, 2]

    return t, x_locs, coords


class StreamingArrowWaveformDataset(IterableDataset):
    def __init__(self, metadata_csv, 
                 input_signals, output_signals, 
                 meta_fields=["subject_id", "hadm_id"],
                 chunk_duration=DEFAULT_CHUNK_DURATION, 
                 forecast_duration=DEFAULT_FORECAST_DURATION, 
                 L=DEFAULT_SPATIAL_LENGTH,
                 M=DEFAULT_M, 
                 validate_chunk_fn=None, shuffle=True, 
                 normalize_t=False,
                 buffer_size=4096):
        self.input_signals = list(input_signals)
        self.output_signals = list(output_signals)
        self.required_signals = self.input_signals + self.output_signals
        self.meta_fields = meta_fields
        self.validate_chunk_fn = validate_chunk_fn or self.default_validate_chunk
        self.chunk_duration = chunk_duration
        self.forecast_duration = forecast_duration
        self.total_duration = chunk_duration + forecast_duration 
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.M = M  # spatial length over L
        self.L = L  # spatial length over L
        self.normalize_t = normalize_t

        df = pd.read_csv(metadata_csv)
        self.paths = [os.path.abspath(row["parquet"]) for _, row in df.iterrows()]
        self.dataset = pds.dataset(self.paths, format="parquet")


    def __iter__(self):
        def chunk_generator():
            fragments = list(self.dataset.get_fragments())
            if self.shuffle:
                random.shuffle(fragments)  # shuffle patients/files

            for fragment in fragments:
                try:
                    scanner = fragment.scanner(columns=self.required_signals + self.meta_fields + ["timestamp", "fs"])
                    for batch in scanner.to_batches():
                        df = pl.from_arrow(batch)
                        if df.height == 0:
                            continue
                        
                        metainfo = {
                            k: df[k][0] for k in self.meta_fields
                        }
                        metainfo["start"] = df["timestamp"][0]
                        metainfo["end"] = df["timestamp"][-1]
                        
                        fs = float(df["fs"][0])
                        dt = 1.0 / fs
                        # filtering:
                        nyquist = 0.5 * fs
                        cutoff = 5 / nyquist #hz - to isolate QRS complex
                        b, a = butter(N=2, Wn=cutoff, btype='high', analog=False)

                        # grab a chunk described by length of input (chunk_duration)
                        chunk_size = int(self.chunk_duration / dt)
                        total_size = int(self.total_duration / dt)
                        # make sure there's enough data ahead
                        if df.height < total_size:
                            continue

                        # partition by chunk_size, but look ahead total_size
                        n_chunks = (df.height - total_size) // chunk_size

                        # coordinate system will be the same for all samples in this fragment (patient)
                        # note: unnomralized
                        t, x_locs, coords = build_coords(dt=dt, T=total_size, M=self.M, L=self.L)
                        t_input = t[:chunk_size]
                        
                        if self.normalize_t:
                            # these are the same number
                            t = t / t[chunk_size - 1].clamp(min=1e-6)
                            t_input = t_input / t_input[-1].clamp(min=1e-6)

                        for i in range(n_chunks):
                            start = i * chunk_size 
                            chunk = df.slice(start, total_size)
                            chunk = chunk.with_columns([
                                pl.col(col).cast(pl.Float32) for col in self.required_signals
                            ])

                            if not self.validate_chunk_fn(chunk):
                                continue
                            
                            #chunk: [------------------ total_size --------------------]
                            #       [---- input_chunk ----] [--- if forecast target ---]

                            input_chunk = chunk.slice(0, chunk_size) # input chunk is first portion of window

                            if "II" in self.input_signals:
                                omega_default = 2 * math.pi * 75 / 60
                                signal = filtfilt(b, a, input_chunk["II"].to_numpy())
                                # window_len = int(fs * 0.5) # rolling median or average
                                # baseline = median_filter(signal, size=window_len)
                                # ecg_detrended = signal - baseline
                                min_rr_interval = int(fs*0.4) # 25 samples
                                prominence = signal.std()
                                peaks, _ = find_peaks(signal, distance=min_rr_interval, prominence=prominence)
                                phi_mask = torch.zeros_like(t)
                                if len(peaks) == 0:
                                    r_peaks = torch.empty(0, dtype=torch.long)
                                    phi_ref = omega_default * t # default 75 bpm
                                else:
                                    r_peaks = torch.tensor(peaks,dtype=torch.long)
                                    if len(peaks) < 2:
                                        phi_ref = omega_default * t
                                    else:
                                        phi_ref = torch.zeros_like(t)
                                        for i in range(len(peaks) - 1):
                                            start, end = peaks[i], peaks[i+1]
                                            duration = end - start 

                                            # time between peaks: linearly increasing phi
                                            local_phi = torch.linspace(0, 2*math.pi, steps=duration)
                                            phi_ref[start:end, 0] = 2 *  math.pi * i + local_phi
                                            phi_mask[start:end, 0] = 1
                                        phi_ref[peaks[-1]:, 0] = 2 * math.pi * (len(peaks) - 1)

                            input = torch.tensor(input_chunk[self.input_signals].to_numpy(), dtype=torch.float32)
                            output = torch.tensor(chunk[self.output_signals].to_numpy(), dtype=torch.float32).squeeze()
                            mask = ~torch.isnan(output)
                            
                            yield {
                                "input": torch.nan_to_num(input, nan=0.0), # shape [B, T, C]
                                "output": torch.nan_to_num(output,nan=0.0), #shape [B, T]
                                "output_mask": mask,  # shape [B, T]
                                "dt": torch.tensor(dt, dtype=torch.float32),
                                "t": t, # torch.tensor(t_scaled, dtype=torch.float32).unsqueeze(1),
                                "t_input": t_input, # torch.tensor(t_scaled, dtype=torch.float32).unsqueeze(1),
                                "x_locs": x_locs,
                                "coords": coords,
                                "r_peaks": r_peaks,
                                "phi_ref": phi_ref,
                                "phi_mask": phi_mask,
                                **metainfo
                                }
                except Exception as e:
                    print(f"⚠️ Skipping {fragment.path}: {e}")

        return self._shuffle_chunks(chunk_generator()) if self.shuffle else chunk_generator()

    def _shuffle_chunks(self, generator):
        buffer = deque(maxlen=self.buffer_size)
        for item in generator:
            buffer.append(item)
            if len(buffer) == self.buffer_size:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                del buffer[idx]
        while buffer:
            yield buffer.popleft()

    def default_validate_chunk(self, chunk_df: pl.DataFrame):
        checks = []

        for col in self.required_signals:
            mask_col = f"{col}_mask"
            has_mask = mask_col in chunk_df.columns
            std_thresh = 3.0 if col in self.output_signals else 1e-6
            # Build Polars expressions for validation
            exprs = [
                (pl.col(col).len() > 0).alias(f"{col}_nonempty"), # column exists
                (pl.col(col).drop_nans().len() / pl.col(col).len() >= 0.9).alias(f"{col}_valid_ratio"), #at least 90% present
                (pl.col(col).drop_nans().unique().len() >= 10).alias(f"{col}_unique"), # 10 unique values
                (~(pl.col(col).std().is_nan()) & (pl.col(col).std() >= std_thresh)).alias(f"{col}_std"), # has plausible variance
            ]

            if col in self.output_signals:  # check map, min, max only for output signals (e.g., ABP)
                exprs.extend([
                    (pl.col(col).drop_nans().mean().is_between(45, 180)).alias(f"{col}_map_valid"), # ABP check
                    # ((pl.col(col).drop_nans().max() - pl.col(col).drop_nans().min()) >= 10).alias(f"{col}_range_ok"),
                    (pl.col(col).drop_nans().min() >= 10).alias(f"{col}_min_ok"),
                    (pl.col(col).drop_nans().max() <= 200).alias(f"{col}_max_ok"),
                ])

            if has_mask:
                exprs.append((pl.col(mask_col).sum() == chunk_df.height).alias(f"{col}_mask"))

            checks.extend(exprs)

        results = chunk_df.select(checks).row(0)
        return all(results)

    def collate_fn(self, batch):
        collated = {}
        keys = batch[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch]
            if (key not in ['r_peaks', "start", "end"] + self.meta_fields) and (isinstance(values[0], torch.Tensor)):
                values = [torch.nan_to_num(v, nan=0.0) for v in values]
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated


# %%
if __name__ == '__main__':
            
    # %% 
    import matplotlib.pyplot as plt 
    import numpy as np 
    from torch.utils.data import DataLoader 
    # %%
    meta_path = "../../data/metainfo_sample.csv"
    meta = pd.read_csv(meta_path)
    sample = meta.iloc[0]
    
    # %%
    def test_loader(meta_path):
        ds = StreamingArrowWaveformDataset(
            metadata_csv=meta_path,
            input_signals=["II", "PLETH"],
            output_signals=["ABP"],
            chunk_duration=4.0,
            forecast_duration=0.0,
            M=64,
            normalize_t=True
        )
        dl = DataLoader(ds, batch_size=1, num_workers=0)

        batch = next(iter(dl))
        print("Sample keys:", batch.keys())
        print("input shape:", batch["input"].shape)
        print("output shape:", batch["output"].shape)

        nan_batches = sum(
            batch["output_mask"].sum() < batch["output_mask"].numel()
            for batch in tqdm(dl, desc="Checking batches for NaNs")
        )
        print(f"{nan_batches} batches contain NaNs")
        return ds
    
    # %%
    dataset = test_loader(meta_path)
    # %%
    def preview_signals(meta_path):
        def _zscore(x):
            mean = np.nanmean(x)
            std = np.nanstd(x)
            if np.isnan(std) or std == 0:
                return np.full_like(x, np.nan)
            return (x - mean) / std

        sample = pd.read_csv(meta_path).iloc[0]
        parquet_path = sample["parquet"]
        df = pl.read_parquet(parquet_path, memory_map=True)
        # chunk = df.slice(0, 249)  # or any valid slice length
        chunk = df.slice(18000, 249)  # or any valid slice length

        t = np.arange(chunk.height)
        lead_ii = chunk["II"].to_numpy()
        pleth = chunk["PLETH"].to_numpy()
        abp = chunk["ABP"].to_numpy()

        fig, ax1 = plt.subplots(1, 1, figsize=(12, 4), sharex=True)

        ax1.plot(t, _zscore(lead_ii), label="ECG (II)", color="black")
        ax1.plot(t, _zscore(pleth), label="PLETH", color="blue")
        ax1.plot(t, _zscore(abp), label="ABP", color="red")
        ax1.set_title("Raw Signals")
        ax1.set_ylabel("Z-score amplitude")
        ax1.set_xlabel("Time (samples)")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
    # %%
    preview_signals(meta_path)


# %%
