import torch
import torchaudio.transforms as T

class EEGToSpectrogram:
    def __init__(self, n_fft=256, hop_length=32, max_freq_bin=30):
        self.n_fft = n_fft
        self.max_freq_bin = max_freq_bin

        # Spectrogram
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )

        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (channels, time)

        # 1. Compute spectrogram
        spec = self.spectrogram(x)  
        # shape: (channels, freq_bins, time_frames)

        # 2. Crop frequency (keep low frequencies only)
        spec = spec[:, :self.max_freq_bin, :]

        # 3. Convert to dB
        spec_db = self.amplitude_to_db(spec)

        # 4. Normalize per sample
        mean = spec_db.mean(dim=[-1, -2], keepdim=True)
        std = spec_db.std(dim=[-1, -2], keepdim=True)
        spec_normalized = (spec_db - mean) / (std + 1e-6)

        return spec_normalized