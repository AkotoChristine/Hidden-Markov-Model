import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import os
import glob

class SensorFeatureExtractor:
    """
    Extract time-domain and frequency-domain features from accelerometer 
    and gyroscope sensor data for Human Activity Recognition
    """
    
    def __init__(self, window_size=50, overlap=0.5):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        window_size : int
            Number of samples per window
        overlap : float
            Overlap ratio between consecutive windows (0 to 1)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
    def load_data(self, filepath):
        """Load sensor data from CSV file"""
        df = pd.read_csv(filepath)
        print(f"Loaded {filepath}: {len(df)} samples")
        return df
    
    def create_windows(self, data):
        """
        Split data into overlapping windows
        
        Returns:
        --------
        windows : list of DataFrames
        """
        windows = []
        start = 0
        
        while start + self.window_size <= len(data):
            window = data.iloc[start:start + self.window_size]
            windows.append(window)
            start += self.step_size
            
        print(f"Created {len(windows)} windows from data")
        return windows
    
    def extract_time_domain_features(self, signal):
        """
        Extract time-domain features from a signal
        
        Parameters:
        -----------
        signal : array-like
            1D signal (e.g., acc_x values)
            
        Returns:
        --------
        features : dict
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = features['max'] - features['min']
        
        # Advanced statistics
        features['median'] = np.median(signal)
        features['mad'] = np.median(np.abs(signal - features['median']))  # Median Absolute Deviation
        features['rms'] = np.sqrt(np.mean(signal**2))  # Root Mean Square
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        
        # Signal energy
        features['energy'] = np.sum(signal**2)
        
        # Zero crossing rate
        features['zcr'] = np.sum(np.diff(np.sign(signal)) != 0)
        
        return features
    
    def extract_frequency_domain_features(self, signal, sampling_rate=100):
        """
        Extract frequency-domain features using FFT
        
        Parameters:
        -----------
        signal : array-like
            1D signal
        sampling_rate : float
            Sampling rate in Hz
            
        Returns:
        --------
        features : dict
        """
        features = {}
        
        # Perform FFT
        n = len(signal)
        fft_values = fft(signal)
        fft_magnitude = np.abs(fft_values[:n//2])
        fft_freqs = fftfreq(n, 1/sampling_rate)[:n//2]
        
        # Spectral features
        features['spectral_energy'] = np.sum(fft_magnitude**2)
        features['spectral_entropy'] = stats.entropy(fft_magnitude + 1e-10)
        
        # Dominant frequency
        if len(fft_magnitude) > 0:
            dominant_idx = np.argmax(fft_magnitude)
            features['dominant_frequency'] = fft_freqs[dominant_idx]
            features['dominant_magnitude'] = fft_magnitude[dominant_idx]
        else:
            features['dominant_frequency'] = 0
            features['dominant_magnitude'] = 0
        
        # Spectral centroid (weighted mean of frequencies)
        if np.sum(fft_magnitude) > 0:
            features['spectral_centroid'] = np.sum(fft_freqs * fft_magnitude) / np.sum(fft_magnitude)
        else:
            features['spectral_centroid'] = 0
            
        return features
    
    def extract_multi_axis_features(self, acc_x, acc_y, acc_z):
        """
        Extract features from multiple axes (accelerometer or gyroscope)
        
        Parameters:
        -----------
        acc_x, acc_y, acc_z : array-like
            3-axis sensor data
            
        Returns:
        --------
        features : dict
        """
        features = {}
        
        # Signal Magnitude Area (SMA)
        features['sma'] = np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
        
        # Signal Magnitude Vector (SMV)
        features['smv'] = np.mean(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2))
        
        # Correlation between axes
        features['corr_xy'] = np.corrcoef(acc_x, acc_y)[0, 1] if len(acc_x) > 1 else 0
        features['corr_xz'] = np.corrcoef(acc_x, acc_z)[0, 1] if len(acc_x) > 1 else 0
        features['corr_yz'] = np.corrcoef(acc_y, acc_z)[0, 1] if len(acc_y) > 1 else 0
        
        return features
    
    def extract_features_from_window(self, window, sampling_rate=100):
        """
        Extract all features from a single window
        
        Parameters:
        -----------
        window : DataFrame
            Window of sensor data
        sampling_rate : float
            Sampling rate in Hz
            
        Returns:
        --------
        features : dict
        """
        features = {}
        
        # Extract accelerometer data
        acc_x = window['acc_x'].values
        acc_y = window['acc_y'].values
        acc_z = window['acc_z'].values
        
        # Extract gyroscope data
        gyr_x = window['gyr_x'].values
        gyr_y = window['gyr_y'].values
        gyr_z = window['gyr_z'].values
        
        # Time-domain features for each axis
        for axis, signal in [('acc_x', acc_x), ('acc_y', acc_y), ('acc_z', acc_z),
                              ('gyr_x', gyr_x), ('gyr_y', gyr_y), ('gyr_z', gyr_z)]:
            time_features = self.extract_time_domain_features(signal)
            for feat_name, feat_value in time_features.items():
                features[f'{axis}_{feat_name}'] = feat_value
        
        # Frequency-domain features for each axis
        for axis, signal in [('acc_x', acc_x), ('acc_y', acc_y), ('acc_z', acc_z),
                              ('gyr_x', gyr_x), ('gyr_y', gyr_y), ('gyr_z', gyr_z)]:
            freq_features = self.extract_frequency_domain_features(signal, sampling_rate)
            for feat_name, feat_value in freq_features.items():
                features[f'{axis}_{feat_name}'] = feat_value
        
        # Multi-axis features
        acc_multi = self.extract_multi_axis_features(acc_x, acc_y, acc_z)
        for feat_name, feat_value in acc_multi.items():
            features[f'acc_{feat_name}'] = feat_value
            
        gyr_multi = self.extract_multi_axis_features(gyr_x, gyr_y, gyr_z)
        for feat_name, feat_value in gyr_multi.items():
            features[f'gyr_{feat_name}'] = feat_value
        
        # Add metadata
        features['activity'] = window['activity'].iloc[0]
        features['subject'] = window['subject'].iloc[0]
        if 'session' in window.columns:
            features['session'] = window['session'].iloc[0]
        
        return features
    
    def process_dataset(self, data_path, sampling_rate=100):
        """
        Process entire dataset and extract features
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file or directory containing CSV files
        sampling_rate : float
            Sampling rate in Hz
            
        Returns:
        --------
        features_df : DataFrame
            DataFrame containing all extracted features
        """
        all_features = []
        
        # Check if path is directory or file
        if os.path.isdir(data_path):
            csv_files = glob.glob(os.path.join(data_path, '*.csv'))
        else:
            csv_files = [data_path]
        
        print(f"Processing {len(csv_files)} file(s)...")
        
        for filepath in csv_files:
            print(f"\nProcessing: {os.path.basename(filepath)}")
            
            # Load data
            data = self.load_data(filepath)
            
            # Create windows
            windows = self.create_windows(data)
            
            # Extract features from each window
            for i, window in enumerate(windows):
                features = self.extract_features_from_window(window, sampling_rate)
                features['window_id'] = i
                features['source_file'] = os.path.basename(filepath)
                all_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"\n{'='*60}")
        print(f"Feature extraction complete!")
        print(f"Total windows processed: {len(features_df)}")
        print(f"Total features extracted: {len(features_df.columns) - 4}")  # Exclude metadata columns
        print(f"Activities: {features_df['activity'].unique()}")
        print(f"{'='*60}")
        
        return features_df
    
    def save_features(self, features_df, output_path='extracted_features.csv'):
        """Save extracted features to CSV file"""
        features_df.to_csv(output_path, index=False)
        print(f"\nFeatures saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize feature extractor
    extractor = SensorFeatureExtractor(window_size=50, overlap=0.5)
    
    # Process your data
    # Option 1: Single file
    features = extractor.process_dataset('combined_test_data_wide.csv', sampling_rate=100)
    
    # Save features
    extractor.save_features(features, 'extracted_test_features.csv')
    
    print("\nFeature Extractor Ready")
    print("Usage:")
    print("  extractor = SensorFeatureExtractor(window_size=50, overlap=0.5)")
    print("  features = extractor.process_dataset('path_to_data.csv', sampling_rate=100)")
    print("  extractor.save_features(features, 'output.csv')")