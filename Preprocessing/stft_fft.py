import librosa
import numpy as np

def compute_frequency_domain_spectrogram(
        audio_path: str,
        sampling_rate: int,
        n_fft: int = 2048,
        hop_length: int = 240,
        win_length: int = 480,
        fmin: int = 15000,
        fmax: int = 19200):
    """
    :param audio_path:
    :param sampling_rate: AMPLITUDE SAMPLES PER SECOND
    :param n_fft:
    :param hop_length: OVERLAP
    :param win_length:
    :param fmin: LOWER FREQUENCY BOUND
    :param fmax: UPPER FREQUENCY BOUND
    """

    if ".wav" not in audio_path:
        print("[ERROR]: 'audio_path' ONLY TAKES AUDIO FILES (.wav)!")
        return

    # Loads audio clip (.wav)
    y, sr = librosa.load(audio_path, sr=sampling_rate)

    # Computes Short-Time Fourier Transform (STFT) --> complex matrix
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Converts to magnitude
    S_mag = np.abs(S)

    # Converts to dB scale if desired
    S_db = librosa.amplitude_to_db(S_mag)

    # Frequency axis for STFT
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)

    # Finds index range for 15k–19.2k
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

    # Crop
    S_ultra = S_db[idx, :]

    return S_ultra.astype(np.float32)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    AUDIO_PATH = "/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/Dataset/audio_segments_10m/100p_10m_seg279.wav"
    SAMPLING_RATE = 48000
    N_FFT = 2048
    HOP_LENGTH = 240
    WIN_LENGTH = 480
    FMIN = 15000
    FMAX = 19200

    s_ultra = compute_frequency_domain_spectrogram(
        audio_path=AUDIO_PATH,
        sampling_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(s_ultra, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title("Ultrasonic Linear-Frequency Spectrogram (15–19.2 kHz)")
    plt.show()