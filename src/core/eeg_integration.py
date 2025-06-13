#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG Integration for NEXARIS Cognitive Load Estimator

This module provides integration with EEG devices for advanced
cognitive load estimation, supporting the NEXARIS NeuroPrint™ feature.
"""

import time
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable

# PyQt imports for signals
from PyQt5.QtCore import QObject, pyqtSignal

# Import utilities
from ..utils.logging_utils import get_logger


class EEGIntegration(QObject):
    """
    Provides integration with EEG devices for cognitive load estimation
    """
    # Define signals for EEG events
    data_received = pyqtSignal(dict)  # EEG data
    cognitive_load_update = pyqtSignal(float)  # Estimated load from EEG
    connection_status_changed = pyqtSignal(bool, str)  # Connected, status message
    
    # EEG frequency bands
    FREQ_BANDS = {
        'delta': (0.5, 4),    # Deep sleep
        'theta': (4, 8),      # Drowsiness, meditation
        'alpha': (8, 13),     # Relaxed awareness
        'beta': (13, 30),     # Active thinking, focus
        'gamma': (30, 100)    # Higher cognitive processes
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EEG integration
        
        Args:
            config: Application configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get EEG configuration
        self.advanced_config = config.get('advanced_features', {})
        self.eeg_config = self.advanced_config.get('eeg', {})
        self.enabled = self.eeg_config.get('enabled', False)
        self.device_type = self.eeg_config.get('device_type', 'none')
        self.device_id = self.eeg_config.get('device_id', '')
        self.sampling_rate = self.eeg_config.get('sampling_rate', 250)  # Hz
        self.channels = self.eeg_config.get('channels', [])
        self.buffer_size = self.eeg_config.get('buffer_size', 5)  # seconds
        
        # Initialize EEG data
        self.reset_data()
        
        # Set up state
        self.is_connected = False
        self.is_recording = False
        self.recording_thread = None
        self.board = None
        
        # Callbacks
        self.data_callbacks = []
        
        # Check if BrainFlow is available
        self.brainflow_available = False
        if self.enabled:
            self._check_brainflow()
        
        self.logger.info("EEG Integration initialized")
    
    def _check_brainflow(self) -> None:
        """
        Check if BrainFlow is available and load it if possible
        """
        try:
            import brainflow
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
            
            # Store BrainFlow modules
            self.brainflow = brainflow
            self.BoardShim = BoardShim
            self.BrainFlowInputParams = BrainFlowInputParams
            self.BoardIds = BoardIds
            self.DataFilter = DataFilter
            self.FilterTypes = FilterTypes
            self.DetrendOperations = DetrendOperations
            
            # Enable debug logging if needed
            if self.config.get('debug', False):
                BoardShim.enable_dev_board_logger()
            
            self.brainflow_available = True
            self.logger.info("BrainFlow is available")
        except ImportError:
            self.logger.warning("BrainFlow is not available. EEG integration will be disabled.")
            self.brainflow_available = False
            self.enabled = False
        except Exception as e:
            self.logger.error(f"Error initializing BrainFlow: {e}")
            self.brainflow_available = False
            self.enabled = False
    
    def reset_data(self) -> None:
        """
        Reset all EEG data
        """
        self.eeg_data = []
        self.band_powers = {}
        self.cognitive_load_estimates = []
        
        # Metrics
        self.current_cognitive_load = 0.5
        self.signal_quality = 0.0
        
        # Data buffer
        self.data_buffer = []
        self.buffer_size_samples = int(self.sampling_rate * self.buffer_size)
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to receive EEG data
        
        Args:
            callback: Function that takes an EEG data dictionary as argument
        """
        self.data_callbacks.append(callback)
        self.logger.debug("Registered data callback")
    
    def _notify_data_callbacks(self, data: Dict[str, Any]) -> None:
        """
        Notify all registered callbacks with EEG data
        
        Args:
            data: EEG data dictionary
        """
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in EEG data callback: {e}")
    
    def connect(self) -> bool:
        """
        Connect to the EEG device
        
        Returns:
            True if connected successfully, False otherwise
        """
        if not self.enabled or not self.brainflow_available:
            self.logger.warning("EEG integration is not enabled or BrainFlow is not available")
            self.connection_status_changed.emit(False, "EEG integration not available")
            return False
        
        if self.is_connected:
            self.logger.warning("Already connected to EEG device")
            return True
        
        try:
            # Map device type to BoardIds
            board_id = self._get_board_id()
            
            # Set up input parameters
            params = self.BrainFlowInputParams()
            params.serial_port = self.device_id
            
            # Create board
            self.board = self.BoardShim(board_id, params)
            
            # Connect to board
            self.board.prepare_session()
            
            self.is_connected = True
            self.logger.info(f"Connected to EEG device: {self.device_type}")
            self.connection_status_changed.emit(True, f"Connected to {self.device_type}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to EEG device: {e}")
            self.connection_status_changed.emit(False, f"Connection error: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """
        Disconnect from the EEG device
        """
        if not self.is_connected or self.board is None:
            return
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        try:
            # Release session
            self.board.release_session()
            self.board = None
            
            self.is_connected = False
            self.logger.info("Disconnected from EEG device")
            self.connection_status_changed.emit(False, "Disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting from EEG device: {e}")
    
    def start_recording(self) -> bool:
        """
        Start recording EEG data
        
        Returns:
            True if recording started successfully, False otherwise
        """
        if not self.is_connected or self.board is None:
            self.logger.warning("Not connected to EEG device")
            return False
        
        if self.is_recording:
            self.logger.warning("Already recording EEG data")
            return True
        
        try:
            # Reset data
            self.reset_data()
            
            # Start streaming
            self.board.start_stream()
            
            # Set up recording state
            self.is_recording = True
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self.logger.info("Started recording EEG data")
            return True
        except Exception as e:
            self.logger.error(f"Error starting EEG recording: {e}")
            return False
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording EEG data and return metrics
        
        Returns:
            Dictionary of EEG metrics
        """
        if not self.is_recording:
            self.logger.warning("Not recording EEG data")
            return self.get_metrics()
        
        try:
            # Stop streaming
            self.board.stop_stream()
            
            # Set up recording state
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
            
            self.logger.info("Stopped recording EEG data")
            
            # Return final metrics
            return self.get_metrics()
        except Exception as e:
            self.logger.error(f"Error stopping EEG recording: {e}")
            return self.get_metrics()
    
    def _recording_loop(self) -> None:
        """
        Background thread for EEG data recording
        """
        update_interval = 0.1  # seconds
        process_interval = 1.0  # seconds
        last_process_time = time.time()
        
        while self.is_recording and self.board is not None:
            try:
                # Get data from board
                data = self.board.get_current_board_data(int(self.sampling_rate * update_interval))
                
                if data.size > 0:
                    # Add to buffer
                    self.data_buffer.append(data)
                    
                    # Trim buffer if too large
                    while len(self.data_buffer) > 0 and \
                          sum(d.shape[1] for d in self.data_buffer) > self.buffer_size_samples:
                        self.data_buffer.pop(0)
                    
                    # Process data periodically
                    current_time = time.time()
                    if current_time - last_process_time >= process_interval:
                        self._process_eeg_data()
                        last_process_time = current_time
            except Exception as e:
                self.logger.error(f"Error in EEG recording loop: {e}")
            
            # Sleep to maintain update interval
            time.sleep(update_interval)
    
    def _process_eeg_data(self) -> None:
        """
        Process EEG data in buffer
        """
        if not self.data_buffer:
            return
        
        try:
            # Concatenate data from buffer
            data = np.concatenate(self.data_buffer, axis=1)
            
            # Get EEG channels
            eeg_channels = self.board.get_eeg_channels(self.board.get_board_id())
            if not eeg_channels:
                return
            
            # Extract EEG data
            eeg_data = data[eeg_channels, :]
            
            # Calculate signal quality
            self.signal_quality = self._calculate_signal_quality(eeg_data)
            
            # Calculate band powers
            self.band_powers = self._calculate_band_powers(eeg_data)
            
            # Estimate cognitive load
            self.current_cognitive_load = self._estimate_cognitive_load(self.band_powers)
            
            # Record timestamp
            timestamp = datetime.now().isoformat()
            
            # Record data
            self.eeg_data.append({
                'timestamp': timestamp,
                'band_powers': self.band_powers.copy(),
                'signal_quality': self.signal_quality
            })
            
            # Record cognitive load estimate
            self.cognitive_load_estimates.append({
                'timestamp': timestamp,
                'cognitive_load': self.current_cognitive_load,
                'signal_quality': self.signal_quality
            })
            
            # Emit signals
            self.data_received.emit({
                'timestamp': timestamp,
                'band_powers': self.band_powers.copy(),
                'signal_quality': self.signal_quality
            })
            
            self.cognitive_load_update.emit(self.current_cognitive_load)
            
            # Notify callbacks
            self._notify_data_callbacks({
                'data_type': 'eeg',
                'timestamp': timestamp,
                'band_powers': self.band_powers.copy(),
                'cognitive_load': self.current_cognitive_load,
                'signal_quality': self.signal_quality
            })
        except Exception as e:
            self.logger.error(f"Error processing EEG data: {e}")
    
    def _calculate_signal_quality(self, eeg_data: np.ndarray) -> float:
        """
        Calculate signal quality from EEG data
        
        Args:
            eeg_data: EEG data array
        
        Returns:
            Signal quality (0.0 to 1.0)
        """
        try:
            # Simple signal quality metric based on variance
            # Low variance could indicate poor connection or flat signal
            variances = np.var(eeg_data, axis=1)
            mean_variance = np.mean(variances)
            
            # Check for NaN or infinite values
            if np.isnan(mean_variance) or np.isinf(mean_variance):
                return 0.0
            
            # Normalize to 0-1 range
            # This is a simple heuristic and may need adjustment
            quality = min(1.0, max(0.0, mean_variance / 100.0))
            
            # Check for very low variance (flat signal)
            if mean_variance < 1.0:
                quality = 0.0
            
            return quality
        except Exception as e:
            self.logger.error(f"Error calculating signal quality: {e}")
            return 0.0
    
    def _calculate_band_powers(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate band powers from EEG data
        
        Args:
            eeg_data: EEG data array
        
        Returns:
            Dictionary of band powers
        """
        try:
            band_powers = {}
            
            # Calculate band powers for each channel and average
            for band_name, (low_freq, high_freq) in self.FREQ_BANDS.items():
                band_power = 0.0
                
                for channel in range(eeg_data.shape[0]):
                    # Get channel data
                    channel_data = eeg_data[channel, :]
                    
                    # Apply detrend
                    self.DataFilter.detrend(channel_data, self.DetrendOperations.LINEAR.value)
                    
                    # Apply bandpass filter
                    self.DataFilter.perform_bandpass(
                        channel_data, self.sampling_rate,
                        low_freq, high_freq, 2,
                        self.FilterTypes.BUTTERWORTH.value, 0
                    )
                    
                    # Calculate power
                    power = np.mean(np.square(channel_data))
                    band_power += power
                
                # Average across channels
                if eeg_data.shape[0] > 0:
                    band_power /= eeg_data.shape[0]
                
                band_powers[band_name] = band_power
            
            # Normalize band powers
            total_power = sum(band_powers.values())
            if total_power > 0:
                for band in band_powers:
                    band_powers[band] /= total_power
            
            return band_powers
        except Exception as e:
            self.logger.error(f"Error calculating band powers: {e}")
            return {band: 0.0 for band in self.FREQ_BANDS}
    
    def _estimate_cognitive_load(self, band_powers: Dict[str, float]) -> float:
        """
        Estimate cognitive load from band powers
        
        Args:
            band_powers: Dictionary of band powers
        
        Returns:
            Estimated cognitive load (0.0 to 1.0)
        """
        try:
            # Simple cognitive load estimation based on band powers
            # Higher beta and gamma with lower alpha indicates higher cognitive load
            if not band_powers:
                return 0.5
            
            # Get band powers
            alpha = band_powers.get('alpha', 0.0)
            beta = band_powers.get('beta', 0.0)
            gamma = band_powers.get('gamma', 0.0)
            theta = band_powers.get('theta', 0.0)
            
            # Calculate cognitive load
            # This is a simple heuristic and may need adjustment
            # Higher beta/alpha ratio indicates higher cognitive load
            if alpha > 0:
                beta_alpha_ratio = beta / alpha
            else:
                beta_alpha_ratio = beta
            
            # Higher gamma indicates higher cognitive processing
            # Higher theta can indicate drowsiness or meditation
            cognitive_load = 0.5 * beta_alpha_ratio + 0.3 * gamma - 0.2 * theta
            
            # Normalize to 0-1 range
            cognitive_load = min(1.0, max(0.0, cognitive_load))
            
            return cognitive_load
        except Exception as e:
            self.logger.error(f"Error estimating cognitive load: {e}")
            return 0.5
    
    def _get_board_id(self) -> int:
        """
        Map device type to BrainFlow board ID
        
        Returns:
            BrainFlow board ID
        """
        # Map device types to BoardIds
        device_map = {
            'muse': self.BoardIds.MUSE_2_BOARD,
            'muse2': self.BoardIds.MUSE_2_BOARD,
            'museS': self.BoardIds.MUSE_S_BOARD,
            'openBCI': self.BoardIds.CYTON_BOARD,
            'ganglion': self.BoardIds.GANGLION_BOARD,
            'synthetic': self.BoardIds.SYNTHETIC_BOARD
        }
        
        # Get board ID from map or default to synthetic
        return device_map.get(self.device_type.lower(), self.BoardIds.SYNTHETIC_BOARD)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current EEG metrics
        
        Returns:
            Dictionary of EEG metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cognitive_load': self.current_cognitive_load,
            'signal_quality': self.signal_quality,
            'band_powers': self.band_powers.copy()
        }
        
        return metrics
    
    def get_available_devices(self) -> List[str]:
        """
        Get list of available EEG devices
        
        Returns:
            List of available device types
        """
        if not self.brainflow_available:
            return []
        
        # Return list of supported devices
        return [
            'muse', 'muse2', 'museS', 'openBCI', 'ganglion', 'synthetic'
        ]
    
    def is_available(self) -> bool:
        """
        Check if EEG integration is available
        
        Returns:
            True if available, False otherwise
        """
        return self.enabled and self.brainflow_available