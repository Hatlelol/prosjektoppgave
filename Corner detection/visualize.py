import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Function to generate a sine wave
def generate_sine_wave(frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return t, np.sin(2 * np.pi * frequency * t)

# Function to plot the Fourier transform
def plot_fourier_transform(signal, sampling_rate):
    n = len(signal)
    k = np.arange(n)
    T = n / sampling_rate
    frq = k / T
    # frq = frq[:n // 2]  # Take only positive frequencies
    Y = fft(signal) / n
    # Y = Y[:n // 2]

    amplitude = np.abs(Y)
    phase = np.angle(Y)

    # Plot amplitude
    plt.subplot(2, 1, 1)
    plt.plot(frq, amplitude)
    plt.title('Fourier Transform of Sine Wave')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    # Plot phase
    plt.subplot(2, 1, 2)
    plt.plot(frq, phase)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    
    # plt.tight_layout()
    plt.show()

# Parameters
frequency = 5  # Frequency of the sine wave in Hz
duration = 1  # Duration of the signal in seconds
sampling_rate = 1000  # Sampling rate in Hz

# Generate sine wave
time, sine_wave = generate_sine_wave(frequency, duration, sampling_rate)

# Plot the sine wave
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time, sine_wave)
plt.title('Sine Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the Fourier transform
plt.subplot(1, 2, 2)
plot_fourier_transform(sine_wave, sampling_rate)