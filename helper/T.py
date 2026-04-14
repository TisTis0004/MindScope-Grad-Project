import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class EEGToGeneralizedST(nn.Module):
    """
    FFT-Optimized Generalized S-Transform.
    Bypasses PyTorch's memory-heavy F.conv1d by using Fast Fourier Transforms (The Convolution Theorem).
    Uses almost zero VRAM and runs incredibly fast.
    """
    def __init__(self, fs=250, min_freq=1, max_freq=40, num_bins=40, time_pool_factor=16, p=1.0):
        super().__init__()
        self.fs = fs
        self.num_bins = num_bins
        self.time_pool_factor = time_pool_factor
        
        freqs = torch.linspace(min_freq, max_freq, num_bins)
        T_window = fs * 10  # 2500 points
        t = torch.arange(-T_window//2, T_window//2) / fs
        
        sigma = p / (2 * torch.pi * freqs)
        
        t_grid = t.unsqueeze(0).repeat(num_bins, 1)
        sigma_grid = sigma.unsqueeze(1).repeat(1, T_window)
        freqs_grid = freqs.unsqueeze(1).repeat(1, T_window)
        
        gaussian_window = torch.exp(- (t_grid ** 2) / (2 * sigma_grid ** 2))
        
        # S-Transform complex conjugate
        complex_wave = torch.exp(-1j * 2 * torch.pi * freqs_grid * t_grid)
        wavelets = gaussian_window * complex_wave
        
        wavelets = wavelets / torch.sum(gaussian_window, dim=1, keepdim=True)
        
        # FFT padding rule for linear convolution: L >= Signal_Length + Kernel_Length - 1
        # 2500 + 2500 - 1 = 4999. We use 5000 for clean math.
        self.n_fft = 5000
        
        # Pre-compute FFT of the wavelets and store on GPU
        wavelets_padded = F.pad(wavelets, (0, self.n_fft - T_window))
        wavelets_fft = torch.fft.fft(wavelets_padded, dim=1)
        
        # Shape: [1, 1, 40, 5000] -> Ready for broadcasting
        self.register_buffer('wavelets_fft', wavelets_fft.view(1, 1, num_bins, self.n_fft))

    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)
            
        B, C, T = x.shape
        
        # Pad the input signal with zeros to n_fft to avoid circular convolution artifacts
        x_padded = F.pad(x, (0, self.n_fft - T))
        
        # 1. Take FFT of the input: [B, C, 5000] -> [B, C, 1, 5000]
        x_fft = torch.fft.fft(x_padded, dim=-1).unsqueeze(2)
        
        # 2. Multiply in the frequency domain (The Convolution Theorem)
        # x_fft: [B, C, 1, 5000] * wavelets_fft: [1, 1, 40, 5000] -> [B, C, 40, 5000]
        out_fft = x_fft * self.wavelets_fft
        
        # 3. Inverse FFT to get back to time domain
        out_complex = torch.fft.ifft(out_fft, dim=-1)
        
        # 4. Crop out the valid "same" convolution region
        # Center crop to get exactly T=2500 points
        start_idx = T // 2
        end_idx = start_idx + T
        out_cropped = out_complex[..., start_idx:end_idx]
        
        # 5. Get magnitude (Absolute value of complex number)
        spectrogram = torch.abs(out_cropped)
        
        # 6. Pool to shrink time dimension for the LSTM
        spectrogram = F.avg_pool2d(spectrogram, kernel_size=(1, self.time_pool_factor))

        if not is_batched:
            spectrogram = spectrogram.squeeze(0)

        return spectrogram


# =========================================================
# NEW AUGMENTATIONS FOR DOMAIN SHIFT ROBUSTNESS
# =========================================================

class SyntheticEMGInjection(nn.Module):
    """
    Simulates realistic EMG (muscle) artifact contamination during training.
    
    WHY THIS MATTERS:
    The TUSZ eval set is flooded with muscle artifacts that your model has
    never seen during training. EMG noise is NOT Gaussian — it's:
      - Broadband, concentrated above ~20Hz
      - Appears as bursts (not continuous)
      - Affects frontal/temporal channels disproportionately
    
    By injecting EMG-like noise during training, the model learns to
    distinguish "broadband EMG noise" from "real seizure patterns" (which
    have specific spectral evolution and spatial distribution).
    """
    def __init__(self, prob: float = 0.4, snr_range: tuple = (0.5, 3.0),
                 burst_fraction_range: tuple = (0.1, 0.5),
                 max_channels_affected: float = 0.5,
                 fs: int = 256):
        super().__init__()
        self.prob = prob
        self.snr_range = snr_range
        self.burst_fraction_range = burst_fraction_range
        self.max_channels_affected = max_channels_affected
        self.fs = fs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, channels, time)"""
        if not self.training:
            return x
        
        B, C, T = x.shape
        device = x.device
        
        for b in range(B):
            if torch.rand(1).item() > self.prob:
                continue
            
            # 1. Decide how many channels to contaminate
            n_affected = max(1, int(C * torch.rand(1).item() * self.max_channels_affected))
            affected_chs = torch.randperm(C, device=device)[:n_affected]
            
            # 2. Generate broadband white noise
            noise = torch.randn(n_affected, T, device=device)
            
            # 3. High-pass filter to simulate EMG's spectral shape (power >20Hz)
            #    Subtract a smoothed version (removes low-freq components)
            kernel_size = max(3, self.fs // 10)  # ~25 samples at 256Hz ≈ removes <10Hz
            if kernel_size % 2 == 0:
                kernel_size += 1
            avg_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
            
            noise_padded = noise.unsqueeze(1)  # [n_affected, 1, T]
            low_freq = F.conv1d(noise_padded, avg_kernel, padding=kernel_size // 2).squeeze(1)
            emg_noise = noise - low_freq  # High-pass filtered noise (EMG-like)
            
            # 4. Create burst mask — EMG is bursty, not continuous
            burst_frac = (self.burst_fraction_range[0] + 
                         torch.rand(1).item() * (self.burst_fraction_range[1] - self.burst_fraction_range[0]))
            burst_len = int(T * burst_frac)
            burst_start = torch.randint(0, max(1, T - burst_len), (1,)).item()
            
            burst_mask = torch.zeros(T, device=device)
            burst_mask[burst_start:burst_start + burst_len] = 1.0
            
            # Smooth the edges for realism (gradual onset/offset)
            edge_len = min(25, burst_len // 4)
            if edge_len > 1:
                burst_mask[burst_start:burst_start + edge_len] = torch.linspace(0, 1, edge_len, device=device)
                burst_end = burst_start + burst_len
                burst_mask[max(0, burst_end - edge_len):burst_end] = torch.linspace(1, 0, edge_len, device=device)
            
            emg_noise = emg_noise * burst_mask.unsqueeze(0)
            
            # 5. Scale to target SNR
            signal_power = x[b, affected_chs].pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            noise_power = emg_noise.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            snr = self.snr_range[0] + torch.rand(1, device=device).item() * (self.snr_range[1] - self.snr_range[0])
            scaled_noise = emg_noise * (signal_power / (noise_power * snr))
            
            # 6. Inject
            x[b, affected_chs] = x[b, affected_chs] + scaled_noise
        
        return x


class RandomAmplitudeRescale(nn.Module):
    """
    Randomly rescales the amplitude of the entire EEG window.
    
    WHY THIS MATTERS:
    The TUSZ eval set has recordings with ~2x the amplitude std of train.
    By randomly rescaling amplitude during training (simulating different  
    recording conditions, electrode impedances, amplifier gains), the model
    becomes invariant to absolute amplitude changes.
    
    Applied BEFORE spectrogram computation so it affects the raw waveform.
    """
    def __init__(self, scale_range: tuple = (0.5, 2.0), prob: float = 0.5):
        super().__init__()
        self.scale_range = scale_range
        self.prob = prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        if torch.rand(1).item() > self.prob:
            return x
        
        # Per-sample random scale factor
        B = x.shape[0]
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(*self.scale_range)
        return x * scale


class RandomTimeShift(nn.Module):
    """
    Randomly shifts the EEG window in time by rolling the signal.
    Simulates slightly different window alignment / seizure onset timing.
    """
    def __init__(self, max_shift_samples: int = 64, prob: float = 0.3):
        super().__init__()
        self.max_shift_samples = max_shift_samples
        self.prob = prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        if torch.rand(1).item() > self.prob:
            return x
        
        shift = torch.randint(-self.max_shift_samples, self.max_shift_samples + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


# =========================================================
# MAIN TRANSFORM: EEGToSpectrogram (UPGRADED)
# =========================================================

class EEGToSpectrogram(nn.Module):
    def __init__(self, n_fft=256, hop_length=32, max_freq_bin=40):
        super().__init__()
        self.n_fft = n_fft
        self.max_freq_bin = max_freq_bin

        # MelSpectrogram focuses logarithmically on the crucial lower frequencies
        self.spectrogram = T.MelSpectrogram(
            sample_rate=256,      # Match the fs used in caching (was 250, now 256)
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=max_freq_bin,  # Directly maps to CNN input expectations (40 bands)
            f_min=0.5,            # Omits massive lower drift noise
            f_max=40.0,           # FIXED: Match bandpass h_freq=40Hz exactly
            power=2.0
        )

        self.amplitude_to_db = T.AmplitudeToDB()

        # SpecAugment: Randomly mask out frequency and time bands
        self.freq_masking = T.FrequencyMasking(freq_mask_param=8)
        self.time_masking = T.TimeMasking(time_mask_param=20)

        # NEW: EEG-specific augmentations — LIGHTENED for from-scratch training
        self.emg_injection = SyntheticEMGInjection(
            prob=0.10,                         # Light: just enough to see EMG patterns
            snr_range=(1.5, 4.0),              # Milder noise levels
            burst_fraction_range=(0.1, 0.3),   # Shorter bursts
            max_channels_affected=0.25,        # Fewer channels affected
            fs=256,
        )
        self.amplitude_rescale = RandomAmplitudeRescale(
            scale_range=(0.85, 1.2),           # Very tight — just mild gain jitter
            prob=0.3,
        )
        self.time_shift = RandomTimeShift(
            max_shift_samples=32,              # ±0.125s (halved)
            prob=0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time)

        if self.training:
            # ====================================================
            # STAGE 1: RAW WAVEFORM AUGMENTATIONS (before spectrogram)
            # Applied in the time domain for maximum realism
            # ====================================================
            
            # 1a. Channel Dropout — light rate for from-scratch training
            dropout_prob = 0.05
            mask_shape = (x.shape[0], x.shape[1], 1)
            mask = (torch.rand(mask_shape, device=x.device) > dropout_prob).float()
            x = x * mask

            # 1b. EMG Injection — inject realistic muscle artifact noise
            # Teaches model to distinguish "broadband noise" from "seizure"
            x = self.emg_injection(x)

            # 1c. Amplitude Rescaling — simulate different recording conditions
            # Makes model invariant to the 2x amplitude shift in eval
            x = self.amplitude_rescale(x)

            # 1d. Time Shift — slight temporal jitter
            x = self.time_shift(x)

        # ====================================================
        # STAGE 2: SPECTROGRAM COMPUTATION (always applied)
        # ====================================================
        
        # 2a. Compute Mel spectrogram
        spec = self.spectrogram(x)
        # shape: (batch, channels, n_mels, time_frames)

        # 2b. Convert to dB scale
        # Clamp power floor to prevent -inf from zeroed-out channels (channel dropout)
        spec = spec.clamp(min=1e-10)
        spec_db = self.amplitude_to_db(spec)
        # Safety clamp: no value below -100 dB (prevents residual -inf)
        spec_db = spec_db.clamp(min=-100.0)

        # 2c. Normalize per-channel
        # The heavy-duty robust normalization is already done at caching time 
        # (Median+IQR on raw signal). Here we just need stable zero-mean unit-var.
        mean = spec_db.mean(dim=[-1, -2], keepdim=True)
        std = spec_db.std(dim=[-1, -2], keepdim=True).clamp(min=1e-4)
        spec_normalized = (spec_db - mean) / std

        # ====================================================
        # STAGE 3: SPECTROGRAM AUGMENTATIONS (training only)
        # ====================================================
        if self.training:
            # 3a. SpecAugment: frequency and time masking
            spec_normalized = self.freq_masking(spec_normalized)
            spec_normalized = self.time_masking(spec_normalized)
            
            # 3b. Gaussian noise on spectrogram (very light)
            noise_factor = 0.03
            noise = torch.randn_like(spec_normalized) * noise_factor
            spec_normalized = spec_normalized + noise

        return spec_normalized