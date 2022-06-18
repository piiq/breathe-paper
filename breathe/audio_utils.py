"""Audio helpers."""
from typing import List, Tuple, Optional
import math

import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image


def trim_and_norm(
    sample_waveform: np.ndarray,
    sample_rate: int = 22050,
    trimmed_seconds: int = 2,
    norm_technique: str = "STD",
    max_deviations: int = 6,
    percentile: int = 95,
):
    """Trim and normalize audio sample.

    Parameters
    ----------
    sample_waveform : np.ndarray
        The sample waveform
    sample_rate : int, optional
        The sampling rate of the audio, by default 22050
    trimmed_seconds : int, optional
        How many seconds to trim at beginning and end (default = 2), by default 2
    norm_technique : str, optional
        The normalization technique ('SDT' or 'PCT') (default = 'SDT'), by default "STD"
    max_deviations : int, optional
        Number of max deviations for 'SDT' normalization (default = 6), by default 6
    percentile : int, optional
        Threshold percentile for 'PCT' normalization (default = 95), by default 95

    Returns
    -------
    np.ndarray
        The resulting sample waveform
    """
    trim_start = int(
        (len(sample_waveform) * trimmed_seconds) / (len(sample_waveform) / sample_rate)
    )
    trim_end = int(len(sample_waveform) - trim_start)
    trimmed_sample_waveform = sample_waveform[trim_start:trim_end]

    if norm_technique == "STD":
        mean_signal = np.mean(trimmed_sample_waveform)
        standard_deviation = np.std(trimmed_sample_waveform)
        clipped_waveform = np.clip(
            trimmed_sample_waveform,
            a_min=mean_signal - max_deviations * standard_deviation,
            a_max=mean_signal + max_deviations * standard_deviation,
        )
    elif norm_technique == "PCT":
        threshold = np.percentile(abs(trimmed_sample_waveform), percentile)
        clipped_waveform = np.clip(
            trimmed_sample_waveform, a_min=(-threshold), a_max=threshold
        )

    normalized = np.max(abs(clipped_waveform))
    norm_clipped_waveform = np.divide(clipped_waveform, normalized)
    return norm_clipped_waveform


def get_5sec_clips(
    sample_waveform: np.ndarray,
    sample_rate: int = 22050,
) -> List:
    """Get the overlapping 5 second clips.

    Parameters
    ----------
    sample_waveform : np.ndarray
        The sample waveform
    sample_rate : int, optional
        The sample rate, by default 22050

    Returns
    -------
    List
        The 5 second clips.
    """
    # Count how many overlapping 5 second clips we can get
    CLIP_LENGTH = 5
    clip_count = int(len(sample_waveform) / sample_rate / CLIP_LENGTH)
    clip_overlaps_count = clip_count - 1
    clip_length = CLIP_LENGTH * sample_rate

    # Divide and select the clips from the middle
    threshold = len(sample_waveform) - (clip_count * clip_length)
    overlaps_threshold = len(sample_waveform) - (clip_overlaps_count * clip_length)

    lower_threshold_id = int(threshold / 2)
    upper_threshold_id = int(len(sample_waveform) - threshold / 2)

    overlaps_lower_threshold_id = int(overlaps_threshold / 2)
    overlaps_upper_threshold_id = int(len(sample_waveform) - overlaps_threshold / 2)

    split_waveform = np.array_split(
        sample_waveform[lower_threshold_id:upper_threshold_id], clip_count
    )

    if clip_overlaps_count > 0:
        split_waveform_overlaps = np.array_split(
            sample_waveform[overlaps_lower_threshold_id:overlaps_upper_threshold_id],
            clip_overlaps_count,
        )
        selected_clips = split_waveform + split_waveform_overlaps
    else:
        selected_clips = split_waveform
    return selected_clips


# pylint: disable=dangerous-default-value
def extract_specs_for_clip(
    clip: np.ndarray,
    sampling_rate: int,
    f_min: int = 0,
    f_max: Optional[int] = None,
    section_lengths: List = [0.025, 0.1, 0.175],
) -> Tuple:
    """Extract spectrograms for file.

    Parameters
    ----------
    clip : np.ndarray
        The clip
    sampling_rate : int
        Sampling rate
    f_min : int
        Lower frequency clipping value
    f_max : int
        Upper frequency clipping value
    section_lengths : List
        FFT section lengths

    Returns
    -------
    Tuple
         Audio clip as tensor, Array of 3 spectrograms
    """
    num_channels = 3

    specs = []
    for i in range(num_channels):
        power_of_2 = round(math.log(section_lengths[i] * sampling_rate, 2))
        fft_section_width = 2**power_of_2

        clip_tensor = torch.Tensor(clip)
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=fft_section_width,
            win_length=fft_section_width,
            hop_length=int(fft_section_width / 2),
            n_mels=128,
            f_min=f_min,
            f_max=f_max,
        )(clip_tensor)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec + eps)
        spec = np.asarray(
            torchvision.transforms.Resize((128, 250))(Image.fromarray(spec))
        )
        specs.append(spec)
    return clip_tensor, specs
