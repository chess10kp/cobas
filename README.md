# Multi-Pulse Ultrasonic Signal Generation & Acoustic Reflection Classification

This project provides a Python script to generate **multi-pulse ultrasonic signals** for acoustic reflection experiments and demonstrates training a **ResNet** model to classify object states (simulating Li-ion battery inspection) using reflected sound from water bottles.

---

## Usage

1. **Configure parameters** in the script (examples):
   - `sample_rate` — sampling rate in Hz  
   - `pulse_duration` — duration of each pulse in seconds  
   - `gap_duration` — gap between pulses in seconds  
   - `total_duration` — total duration of the generated signal in seconds  
   - `start_freq` — start frequency of sweep in Hz (e.g., 15000)  
   - `end_freq` — end frequency of sweep in Hz (e.g., 19000)  
   - `amplitude` — pulse amplitude (0.0 to 1.0)  
   - `playback_delay` — silent delay after signal playback in seconds

2. **Run the script**:  
   `python ultrasonic_signal.py`

3. **Generated outputs**:
   - WAV file: `ultrasonic_multi_pulse_15s_with_delay.wav`  
   - Waveform plot (Matplotlib) for visual inspection  
   - Spectrogram image(s) used as ResNet input  
   - Trained ResNet model file (e.g., `resnet_model.pth`) — optional, depending on your training code

---

## Experimental Setup

- **Signal source:** MacBook speaker (plays the generated WAV file).  
- **Targets:** Water bottles (full, half-full, empty).  
- **Placement:** Bottles placed **10–15 cm** to the right of the MacBook speaker.  
- **Recording device:** iPhone microphone placed close to the bottle; recorded at **different angles** (moved the iPhone relative to the bottle to capture angle-dependent reflections).  
- **Procedure (per recording):**
  1. Place bottle 10–15 cm right of MacBook speaker.  
  2. Position iPhone mic close to the bottle at chosen angle.  
  3. Start recording on iPhone.  
  4. Play the WAV file from the MacBook speaker.  
  5. Stop recording after the playback + delay.  
  6. Repeat for other angles and other bottle fill states.  
- **Dataset:** ~10–13 recordings per class (full, half, empty) used for training; additional held-out recordings used for testing.  
- **Model:** Convert recordings to spectrograms, train a ResNet (image-classification style) on those spectrograms to classify bottle states.

---


## Output

### Generated files
- `ultrasonic_multi_pulse_15s_with_delay.wav` — the generated multi-pulse ultrasonic signal.  
- `waveform_plot.png` (or inline Matplotlib figure) — visualization of generated time-domain waveform.  
- `resnet_model.pth` — (example) saved ResNet model weights after training.

### Model training & example results
- **Training data:** 10–13 samples per class (full / half / empty).  
- **Testing:** Held-out recordings recorded at different iPhone angles.  
- **Performance:** Good and consistent classification accuracy observed (example below is illustrative).


**Processing pipeline (high level):**  
Play WAV (MacBook) → Capture reflected audio (iPhone) → Convert audio → Compute spectrogram → Train/Infer with ResNet

---

## Author

Shamit — research in acoustic & optical signal processing for battery-health detection
