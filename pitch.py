from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def __check_yin_params(
    *, sr: float, fmax: float, fmin: float, frame_length: int, win_length: int
):
    """Check the feasibility of yin/pyin parameters against
    the following conditions:

    1. 0 < fmin < fmax <= sr/2
    2. frame_length - win_length - 1 > sr/fmax
    """
    if fmax > sr / 2:
        raise ValueError(f"fmax={fmax:.3f} cannot exceed Nyquist frequency {sr/2}")
    if fmin >= fmax:
        raise ValueError(f"fmin={fmin:.3f} must be less than fmax={fmax:.3f}")
    if fmin <= 0:
        raise ValueError(f"fmin={fmin:.3f} must be strictly positive")

    if win_length >= frame_length:
        raise ValueError(
            f"win_length={win_length} must be less than frame_length={frame_length}"
        )

    if frame_length - win_length - 1 <= sr // fmax:
        fmax_feasible = sr / (frame_length - win_length - 1)
        frame_length_feasible = int(np.ceil(sr/fmax) + win_length + 1)
        raise ValueError(
            f"fmax={fmax:.3f} is too small for frame_length={frame_length}, win_length={win_length}, and sr={sr}. "
            f"Either increase to fmax={fmax_feasible:.3f} or frame_length={frame_length_feasible}"
        )
    
def _cumulative_mean_normalized_difference(
    y_frames: torch.Tensor,
    frame_length: int,
    win_length: int,
    min_period: int,
    max_period: int,
) -> torch.Tensor:
    # Autocorrelation.
    a = torch.fft.rfft(y_frames, frame_length, dim=-1)
    b = torch.fft.rfft(y_frames[..., torch.arange(win_length, 0, -1)], frame_length, dim=-1)
    acf_frames = torch.fft.irfft(a * b, frame_length, dim=-1)[..., win_length:]
    acf_frames[torch.abs(acf_frames) < 1e-6] = 0

    # Energy terms.
    energy_frames = (y_frames**2).cumsum(-1)
    energy_frames = (
        energy_frames[..., win_length:] - energy_frames[..., :-win_length]
    )
    energy_frames[torch.abs(energy_frames) < 1e-6] = 0

    # Difference function.
    yin_frames = energy_frames[..., :1] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[..., min_period : max_period + 1]
    # broadcast this shape to have leading ones
    sp = [1 for _ in range(yin_frames.dim())]
    sp[-1] = max_period
    tau_range = torch.broadcast_to(torch.arange(1, max_period + 1), sp)

    cumulative_mean = (
        yin_frames[..., 1 : max_period + 1].cumsum(-1) / tau_range
    )
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period]
    yin_frames: torch.Tensor = yin_numerator / (
        yin_denominator + torch.finfo(yin_denominator.dtype).tiny
    )
    return yin_frames

def _pi_stencil(x: torch.Tensor, i: int) -> torch.Tensor:
    a = x[..., i + 1] + x[..., i - 1] - 2 * x[..., i]
    b = (x[..., i + 1] - x[..., i - 1]) / 2

    return torch.where(b.abs() < a.abs(), -b / a, 0.)

def _pi_wrapper(x: torch.Tensor, y: torch.Tensor) -> None:  # pragma: no cover
    for i in range(1, x.shape[-1] - 1):
        y[:, i] = _pi_stencil(x, i)

def _parabolic_interpolation(x: torch.Tensor) -> torch.Tensor:
    # Allocate the output array and rotate target axis
    shifts = torch.empty_like(x)

    # Call the vectorized stencil
    _pi_wrapper(x, shifts)

    # Handle the edge condition not covered by the stencil
    shifts[..., -1] = 0
    shifts[..., 0] = 0

    return shifts

def _localmin_sten(x, i):  # pragma: no cover
    return (x[..., i] < x[..., i - 1]) & (x[..., i] <= x[..., i + 1])

def _localmin(x: torch.Tensor, y: torch.Tensor) -> None:  # pragma: no cover
    for i in range(1, x.shape[-1] - 1):
        y[..., i] = _localmin_sten(x, i)

def localmin(x: torch.Tensor) -> torch.Tensor:
    # Allocate the output array and rotate target axis
    lmin = torch.empty_like(x, dtype=bool)

    # Call the vectorized stencil
    _localmin(x, lmin)

    # Handle the edge condition not covered by the stencil
    lmin[..., -1] = x[..., -1] < x[..., -2]

    return lmin

def yin(
    y: torch.Tensor,
    *,
    fmin: float,
    fmax: float,
    sr: float = 22050,
    frame_length: int = 2048,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    trough_threshold: float = 0.1
) -> torch.Tensor:
    """Fundamental frequency (F0) estimation using the YIN algorithm.

    YIN is an autocorrelation based method for fundamental frequency estimation [#]_.
    First, a normalized difference function is computed over short (overlapping) frames of audio.
    Next, the first minimum in the difference function below ``trough_threshold`` is selected as
    an estimate of the signal's period.
    Finally, the estimated period is refined using parabolic interpolation before converting
    into the corresponding frequency.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Args:
        y (torch.Tensor): 
            audio time series.
        fmin (float): number > 0 [scalar]
            minimum frequency in Hertz.
            The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
            though lower values may be feasible.
        fmax (float): number > fmin, <= sr/2 [scalar]
            maximum frequency in Hertz.
            The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
            though higher values may be feasible.
        sr (float, optional): 
            sampling rate of ``y`` in Hertz.
            Defaults to 22050.
        frame_length (int, optional): int > 0 [scalar]
            length of the frames in samples.
            By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
            a sampling rate of 22050 Hz. 
            Defaults to 2048.
        win_length (Optional[int], optional): 
            length of the window for calculating autocorrelation in samples.
            If ``None``, defaults to ``frame_length // 2``.
        hop_length (Optional[int], optional): 
            number of audio samples between adjacent YIN predictions.
            If ``None``, defaults to ``frame_length // 4``.
        trough_threshold (float, optional): number > 0 [scalar]
            absolute threshold for peak estimation.
            Defaults to 0.1.

    Returns:
        torch.Tensor: time series of fundamental frequencies in Hertz.
    """
    if fmin is None or fmax is None:
        raise ValueError('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    __check_yin_params(
        sr=sr, fmax=fmax, fmin=fmin, frame_length=frame_length, win_length=win_length
    )

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4
    
    # frame the audio 
    # note: i give up the center option here (https://github.com/librosa/librosa/blob/cd66e10edf5fad2e95267b8289463848d54c39b7/librosa/core/pitch.py#L590) #noqa
    # instead only pad it right to frame_length when the audio is too short 
    if y.shape[-1] < frame_length:
        y = F.pad(y, [0, frame_length - y.shape[-1]])
    y_frames = y.unfold(dimension=-1, size=frame_length, step=hop_length)

    # Calculate minimum and maximum periods
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = localmin(yin_frames)
    is_trough[..., 0] = yin_frames[..., 0] < yin_frames[..., 1]

    # Find minima below peak threshold.
    is_threshold_trough = torch.logical_and(is_trough, yin_frames < trough_threshold)

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    target_shape = list(yin_frames.shape)
    target_shape[-1] = 1

    global_min = torch.argmin(yin_frames, dim=-1, keepdim=True)
    yin_period = torch.argmax(is_threshold_trough.to(torch.long), dim=-1, keepdim=True)

    no_trough_below_threshold = torch.all(~is_threshold_trough, dim=-1, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.

    yin_period = (
        min_period
        + yin_period
        + torch.take_along_dim(parabolic_shifts, yin_period, dim=-1)
    )[..., 0]

    # Convert period to fundamental frequency.
    f0: torch.Tensor = sr / yin_period
    return f0


if __name__ == "__main__":
    import torchaudio
    wave, sr = torchaudio.load("./data/data_aishell3/test/wav/SSB1831/SSB18310007.wav")
    if sr != 8000:
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=8000)
        sr = 8000

    wave.requires_grad_()
    f0 = yin(wave.squeeze(0), fmin=20, fmax=2000, sr=8000)

    loss = F.mse_loss(f0, torch.zeros_like(f0))
    loss.backward()

    print("".format(loss.item()))
    print("Gradient wave sum: " + str(wave.grad.sum()))
    print(f0)
