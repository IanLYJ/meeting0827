import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

def generate_periodic_signal(duration=2.0, sampling_rate=1000):
    """
    Generate a periodic signal as combination of sine waves
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    
    # Combine multiple sine waves with different frequencies and amplitudes
    freq1, freq2, freq3 = 10, 16, 24  # Hz
    amp1, amp2, amp3 = 1.0, 0.5, 0.5
    
    signal_clean = (amp1 * np.sin(2 * np.pi * freq1 * t) + 
                   amp2 * np.sin(2 * np.pi * freq2 * t) + 
                   amp3 * np.sin(2 * np.pi * freq3 * t))
    
    return t, signal_clean

def add_noise(signal_clean, noise_level=0.3):
    """
    Add white noise to the signal
    """
    noise = np.random.normal(0, noise_level, len(signal_clean))
    signal_noisy = signal_clean + noise
    return signal_noisy

def apply_moving_average(signal_noisy, window_size=21):
    """
    Apply moving average filter to smooth the data
    """
    # Create a uniform kernel for moving average
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution with 'same' mode to keep original signal length
    signal_filtered = np.convolve(signal_noisy, kernel, mode='same')
    
    return signal_filtered


def apply_butterworth_filter(signal_noisy, sampling_rate, cutoff_freq=30, order=4):
    """
    Apply Butterworth low-pass filter to smooth the data
    """
    # Normalize the cutoff frequency (0 to 1, where 1 is Nyquist frequency)
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    signal_filtered = signal.filtfilt(b, a, signal_noisy)
    
    return signal_filtered

def find_main_frequencies(signal_data, sampling_rate, num_peaks=3):
    """
    Apply FFT and find the main frequencies
    """
    # Compute FFT
    n = len(signal_data)
    yf = fft(signal_data)
    xf = fftfreq(n, 1/sampling_rate)
    
    # Take only positive frequencies
    positive_freqs = xf[:n//2]
    magnitude = np.abs(yf[:n//2])
    
    # Find peaks (main frequencies)
    peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
    
    # Sort by magnitude and get top frequencies
    peak_magnitudes = magnitude[peaks]
    sorted_indices = np.argsort(peak_magnitudes)[::-1]
    top_peaks = peaks[sorted_indices[:num_peaks]]
    
    main_frequencies = positive_freqs[top_peaks]
    main_magnitudes = magnitude[top_peaks]
    
    return xf, yf, main_frequencies, main_magnitudes

def main():
    # Parameters
    duration = 5.0  # seconds
    sampling_rate = 1000  # Hz
    noise_level = 0.5
    cutoff_freq = 30  # Hz for Butterworth filter
    window_size = 21
    
    # Step 1: Generate periodic signal with combination of sine waves
    print("Step 1: Generating periodic signal...")
    t, signal_clean = generate_periodic_signal(duration, sampling_rate)
    
    # Step 2: Add noise
    print("Step 2: Adding noise...")
    signal_noisy = add_noise(signal_clean, noise_level)
    
    # Step 3: Apply Butterworth filter
    print("Step 3: Applying Butterworth filter...")

    # signal_filtered = apply_butterworth_filter(signal_noisy, sampling_rate, cutoff_freq)
    signal_filtered = apply_moving_average(signal_noisy, window_size)
    
    # Step 4: Apply FFT to find main frequencies
    print("Step 4: Applying FFT to find main frequencies...")
    xf, yf, main_frequencies, main_magnitudes = find_main_frequencies(signal_filtered, sampling_rate)
    
    # Display results
    print("\nMain frequencies found:")
    for i, (freq, mag) in enumerate(zip(main_frequencies, main_magnitudes)):
        print(f"Frequency {i+1}: {freq:.2f} Hz (Magnitude: {mag:.2f})")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Original clean signal
    ax1.plot(t, signal_clean, 'b-', linewidth=1)
    ax1.set_title('Step 1: Original Clean Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot 2: Noisy signal
    ax2.plot(t, signal_noisy, 'r-', alpha=0.7, linewidth=0.8)
    ax2.plot(t, signal_clean, 'b--', alpha=0.8, linewidth=1, label='Original')
    ax2.set_title('Step 2: Signal with Added Noise')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Filtered signal
    ax3.plot(t, signal_filtered, 'g-', linewidth=1, label='Filtered')
    ax3.plot(t, signal_clean, 'b--', alpha=0.6, linewidth=1, label='Original')
    ax3.set_title('Step 3:Filtered Signal')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: FFT result
    positive_freqs = xf[:len(xf)//2]
    magnitude = np.abs(yf[:len(yf)//2])
    ax4.plot(positive_freqs, magnitude, 'purple', linewidth=1)
    ax4.scatter(main_frequencies, main_magnitudes, color='red', s=100, zorder=5, 
               label=f'Main frequencies: {main_frequencies[:3]}')
    ax4.set_title('Step 4: FFT - Frequency Domain')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim(0, 50)  # Focus on lower frequencies
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return signal_clean, signal_noisy, signal_filtered, main_frequencies

if __name__ == "__main__":
    # Run the main function
    signal_clean, signal_noisy, signal_filtered, frequencies = main()