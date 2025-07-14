import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import StreamingArrowWaveformDataset

class WaveformDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_signals,
        output_signals,
        batch_size = 8,
        buffer_size=8192,
        num_workers = 0,
        chunk_duration = 4.0,
        forecast_duration = 0.0,
        normalize_t = True,
        train_csv = None,
        val_csv = None,
        test_csv = None,
        val_batch_size = None,
        test_batch_size = None,
        val_buffer_size=None,
        test_buffer_size=None,
        M = 64,
        L = 0.25,
        shuffle = True,
        pin_memory = True,
    ):
        super().__init__()
        self.train_csv = train_csv 
        self.val_csv = val_csv 
        self.test_csv = test_csv 
        self.train_batch_size = batch_size 
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size 
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size 
        self.train_buffer_size = buffer_size 
        self.val_buffer_size = val_buffer_size if val_buffer_size is not None else buffer_size 
        self.test_buffer_size = test_buffer_size if test_buffer_size is not None else buffer_size 

        self.input_signals = input_signals
        self.output_signals = output_signals
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_duration = chunk_duration
        self.forecast_duration = forecast_duration
        self.normalize_t = normalize_t
        self.M = M
        self.L = L
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if self.train_csv:
            self.train_dataset = StreamingArrowWaveformDataset(
                metadata_csv=self.train_csv,
                input_signals=self.input_signals,
                output_signals=self.output_signals,
                chunk_duration=self.chunk_duration,
                forecast_duration=self.forecast_duration,
                normalize_t=self.normalize_t,
                M=self.M,
                L=self.L,
                shuffle=self.shuffle,
                buffer_size=self.train_buffer_size
            )
        if self.val_csv:
            self.val_dataset = StreamingArrowWaveformDataset(
                metadata_csv=self.val_csv,
                input_signals=self.input_signals,
                output_signals=self.output_signals,
                chunk_duration=self.chunk_duration,
                forecast_duration=self.forecast_duration,
                normalize_t=self.normalize_t,
                M=self.M,
                L=self.L,
                shuffle=False,
                buffer_size=self.val_buffer_size
            )
        if self.test_csv:
            self.test_dataset = StreamingArrowWaveformDataset(
                metadata_csv=self.test_csv,
                input_signals=self.input_signals,
                output_signals=self.output_signals,
                chunk_duration=self.chunk_duration,
                forecast_duration=self.forecast_duration,
                normalize_t=self.normalize_t,
                M=self.M,
                L=self.L,
                shuffle=False,
                buffer_size=self.test_buffer_size
            )
        # You could define val_dataset here too (same class, but shuffle=False)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False
        )