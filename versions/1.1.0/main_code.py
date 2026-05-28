#updated fall logic

import subprocess
import threading
import time
import glob
import os
import csv
import json
import queue
import qrcode
import configparser
import random
import psutil  # Added for system stats
from datetime import datetime
from collections import deque
import paho.mqtt.client as mqtt
import logging
from pathlib import Path

# --- NEW IMPORTS FOR OTA & GUI ---
import sys
import argparse
import requests
import zipfile
import hashlib

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor
# --- END NEW IMPORTS ---

# === APPLICATION VERSION ===
# Version is passed as a command-line argument

# === LOGGING CONFIG ===
# Logging is configured in main, as we need the shared_data_dir
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Temporary logger initialized. File logging will start after config.")


# === CONFIG MANAGEMENT ===
class DeviceConfig:
    """Manages device configuration with persistence"""
    def __init__(self, shared_data_dir):
        self.shared_data_dir = shared_data_dir
        self.config_file = os.path.join(self.shared_data_dir, "device_config.ini")
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            logger.info(f"No config file found at {self.config_file}. Creating default.")
            self.create_default_config()

        # Update paths to match TODAY's date
        self.update_paths()
        
        # Save the corrected paths back to the file
        self.save_config()

    def update_paths(self):
        """Ensure paths are relative to the current shared_data_dir and include TODAY'S date. Creates files if they don't exist."""
        if 'PATHS' not in self.config:
            self.config['PATHS'] = {}

        # Get current date string for filenames
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Define the correct structure with Date appended
        updates = {
            'output_path': os.path.join(self.shared_data_dir, 'logs', 'unified_log', f'unified_log_{date_str}.jsonl'),
            'heartbeat_log': os.path.join(self.shared_data_dir, 'logs', f'heartbeat_payload_{date_str}.csv'),
            'novelda_bpm': os.path.join(self.shared_data_dir, 'logs', 'Novelda_Data', f'bpm_{date_str}.csv'),
            'novelda_dis': os.path.join(self.shared_data_dir, 'logs', 'Novelda_Data', f'distance_{date_str}.csv'),
            'novelda_raw': os.path.join(self.shared_data_dir, 'logs', 'Novelda_Data', f'radar_data_log_{date_str}.csv'),
            'ti_raw_log': os.path.join(self.shared_data_dir, 'logs', 'TI_DATA', f'radar_raw_log_{date_str}.csv'),
            'ti_output_log': os.path.join(self.shared_data_dir, 'logs', 'TI_DATA', f'radar_output_log_{date_str}.csv'),
            'fusion_log': os.path.join(self.shared_data_dir, 'logs', f'fusion_data_{date_str}.json'),
            # Added num_people path (static filename as per request)
            'num_people': os.path.join(self.shared_data_dir, 'logs', 'num_people.csv'),
        }

        # Apply updates and ensure files exist
        for key, path in updates.items():
            self.config['PATHS'][key] = path
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create file if it doesn't exist
            if not os.path.exists(path):
                try:
                    with open(path, 'a') as f:
                        # Optional: write headers for CSV files
                        if path.endswith('.csv'):
                            if key == 'heartbeat_log':
                                f.write("timestamp,device_id,uptime_seconds,version,system_stats,last_recording_status,error_codes\n")
                            elif key == 'num_people':
                                f.write("Timestamp,Person_Count\n")
                            elif key in ['novelda_bpm', 'novelda_dis', 'novelda_raw', 'ti_raw_log', 'ti_output_log']:
                                f.write("timestamp,value\n") # Generic header to prevent parser errors
                    logger.info(f"Created new daily file: {path}")
                except Exception as e:
                    logger.error(f"Could not initialize file {path}: {e}")

    def create_default_config(self):
        """Create default configuration"""
        self.config['MQTT'] = {
            'broker': 'broker.address',
            'port': '8883',
            'tenant_id': '1',
            'device_id': 'device001',
            'use_tls': 'true',
            'username': '',
            'password': ''
        }
        self.config['TLS'] = {
            'ca_cert': os.path.join(self.shared_data_dir, 'certs', 'rootCA.pem'),
        }
        # Paths are handled by update_paths() called in load_config
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                self.config.write(f)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, section, key, fallback=None):
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def set(self, section, key, value):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value) # Ensure value is string for configparser
        self.save_config()
    
    def generate_qr_code(self, output_file="device_config_qr.png"):
        """Generate QR code for device configuration"""
        qr_output_path = os.path.join(self.shared_data_dir, output_file)
        config_data = {
            'broker': self.get('MQTT', 'broker'),
            'port': self.get('MQTT', 'port'),
            'tenant_id': self.get('MQTT', 'tenant_id'),
            'device_id': self.get('MQTT', 'device_id')
        }
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(config_data))
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img.save(qr_output_path)
        logger.debug(f"QR code saved to {qr_output_path}")
    
    def load_from_qr_data(self, qr_data):
        """Load configuration from QR code data"""
        try:
            data = json.loads(qr_data)
            self.set('MQTT', 'broker', data.get('broker', self.get('MQTT', 'broker')))
            self.set('MQTT', 'port', str(data.get('port', self.get('MQTT', 'port'))))
            self.set('MQTT', 'tenant_id', data.get('tenant_id', self.get('MQTT', 'tenant_id')))
            self.set('MQTT', 'device_id', data.get('device_id', self.get('MQTT', 'device_id')))
            logger.debug("Configuration loaded from QR code")
            return True
        except Exception as e:
            logger.error(f"Failed to load QR config: {e}")
            return False

# === MESSAGE BUFFER ===
class MessageBuffer:
    """Persistent message buffer for offline operation"""
    def __init__(self, shared_data_dir, max_size=1000):
        self.buffer_file = os.path.join(shared_data_dir, "message_buffer.json")
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.offline_start = None
        self.total_offline_duration = 0
        self.load_buffer()
    
    def load_buffer(self):
        """Load buffered messages from file"""
        try:
            if os.path.exists(self.buffer_file):
                with open(self.buffer_file, 'r') as f:
                    messages = json.load(f)
                    self.buffer.extend(messages[-self.max_size:])
                logger.debug(f"Loaded {len(self.buffer)} buffered messages from previous session")
                
                # Calculate offline duration from oldest message
                if self.buffer:
                    oldest_msg = self.buffer[0]
                    offline_duration = time.time() - oldest_msg['timestamp']
                    logger.debug(f"System was offline for approximately {offline_duration/60:.1f} minutes")
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")
    
    def save_buffer(self):
        """Save buffer to file"""
        try:
            os.makedirs(os.path.dirname(self.buffer_file), exist_ok=True)
            with self.lock:
                with open(self.buffer_file, 'w') as f:
                    json.dump(list(self.buffer), f)
        except Exception as e:
            logger.error(f"Failed to save buffer: {e}")
    
    def add_message(self, topic, payload, qos=0):
        """Add message to buffer"""
        with self.lock:
            # Track when we went offline
            if len(self.buffer) == 0 and self.offline_start is None:
                self.offline_start = time.time()
                
            message = {
                'topic': topic,
                'payload': payload,
                'qos': qos,
                'timestamp': time.time()
            }
            self.buffer.append(message)
            self.save_buffer()
    
    def get_messages(self, limit=None):
        """Get buffered messages"""
        with self.lock:
            if limit:
                return list(self.buffer)[:limit]
            return list(self.buffer)
    
    def remove_messages(self, count):
        """Remove successfully sent messages"""
        with self.lock:
            for _ in range(min(count, len(self.buffer))):
                self.buffer.popleft()
            
            # If buffer is now empty, calculate offline duration
            if len(self.buffer) == 0 and self.offline_start:
                offline_duration = time.time() - self.offline_start
                self.total_offline_duration += offline_duration
                logger.debug(f"Back online after {offline_duration/60:.1f} minutes offline")
                self.offline_start = None
                
            self.save_buffer()
    
    def clear(self):
        """Clear all buffered messages"""
        with self.lock:
            self.buffer.clear()
            self.save_buffer()
            self.offline_start = None

# === TI RADAR PROCESS MONITOR ===
class TIRadarProcessMonitor:
    """Monitors TI Radar process stderr for errors."""
    def __init__(self):
        self.error_thread = None
        
    def start_reading(self, process):
        """Start reading from process stderr"""
        self.error_thread = threading.Thread(target=self._read_errors, args=(process,))
        self.error_thread.daemon = True
        self.error_thread.start()
            
    def _read_errors(self, process):
        """Read stderr for debugging"""
        try:
            while process.poll() is None:
                line = process.stderr.readline()
                if line:
                    line = line.strip()
                    if line:  # Only print non-empty lines
                        logger.debug(f"TI STDERR: {line}")
        except Exception as e:
            logger.error(f"Error reading TI Radar stderr: {e}")

# === BASE RADAR CSV MONITOR ===
class BaseRadarCsvMonitor:
    """
    Base class to monitor a CSV file for new lines and health.
    Handles dynamic file switching if path changes (daily rotation).
    """
    def __init__(self, csv_path, stale_threshold_seconds=60):
        self.csv_path = csv_path
        self.monitor_thread = None
        self.last_file_pos = 0
        # Initialize to current time to avoid staleness check on old files at startup
        self.last_file_mod_time = time.time() 
        self.stale_threshold_seconds = stale_threshold_seconds
        self.last_stale_warning_time = 0
        self._is_healthy = True # Assume healthy initially to allow processes to boot
        self.health_lock = threading.RLock()
        
        # Initialize file position to end of file to skip old history
        if os.path.exists(self.csv_path):
            try:
                self.last_file_pos = os.path.getsize(self.csv_path)
            except Exception:
                self.last_file_pos = 0

    def start_monitoring(self):
        """Start monitoring the CSV file in a separate thread."""
        logger.info(f"Starting CSV monitor on: {self.csv_path}")
        # Reset trackers for the new monitoring session
        self.last_file_mod_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def update_path(self, new_path):
        """Called when the date changes to switch to the new daily file"""
        if new_path != self.csv_path:
            logger.info(f"Switching CSV monitor from {self.csv_path} to {new_path}")
            self.csv_path = new_path
            self.last_file_pos = 0 # New file, start from beginning
            self.last_file_mod_time = time.time()

    def _monitor_loop(self):
        """
        Continuously reads the CSV file for new lines.
        """
        time.sleep(2) # Give file a moment to be created
        
        while True:
            try:
                self._check_csv_staleness()

                if os.path.exists(self.csv_path):
                    with open(self.csv_path, 'r') as f:
                        f.seek(self.last_file_pos)
                        new_lines = f.readlines()
                        
                        if new_lines:
                            for line in new_lines:
                                line = line.strip()
                                if not line or "timestamp" in line:
                                    continue
                                self._parse_csv_line(line)
                            
                            self.last_file_pos = f.tell()
                            # Successfully read new lines, update mod tracker
                            self.last_file_mod_time = time.time()
                        
            except FileNotFoundError:
                logger.debug(f"CSV monitor: file not found yet, waiting... ({self.csv_path})")
            except Exception as e:
                logger.error(f"Error in CSV monitor loop ({self.csv_path}): {e}")
            
            time.sleep(1.0) 

    def _check_csv_staleness(self):
        """Checks if the CSV file has stopped updating."""
        try:
            if not os.path.exists(self.csv_path):
                # If file doesn't exist yet, we only flag as unhealthy if we've waited too long
                if time.time() - self.last_file_mod_time > self.stale_threshold_seconds:
                    with self.health_lock:
                        self._is_healthy = False
                return

            # Check if the file's ACTUAL modification time is newer than our last read
            current_mod_time = os.path.getmtime(self.csv_path)
            
            # If the file hasn't changed on disk, check the age
            if current_mod_time <= self.last_file_mod_time:
                age = time.time() - self.last_file_mod_time
                if age > self.stale_threshold_seconds:
                    with self.health_lock:
                        self._is_healthy = False
                    if time.time() - self.last_stale_warning_time > 60:
                        logger.error(f"CSV file is stale! No updates in {age:.0f} seconds: {self.csv_path}")
                        self.last_stale_warning_time = time.time()
            else:
                with self.health_lock:
                    self._is_healthy = True
                # Note: We update last_file_mod_time to the file's time
                self.last_file_mod_time = current_mod_time
                self.last_stale_warning_time = 0 

        except Exception as e:
            logger.warning(f"Error checking CSV staleness ({self.csv_path}): {e}")

    def _parse_csv_line(self, line):
        """Placeholder for child classes to implement."""
        raise NotImplementedError

    def is_healthy(self):
        """Thread-safe method to check health status."""
        with self.health_lock:
            return self._is_healthy

# === TI RADAR ACTIVITY CSV READER ===
class TIActivityReader(BaseRadarCsvMonitor):
    """
    Monitors the TI Radar output CSV file in real-time.
    Supports multi-person tracking (Person 1 and Person 2).
    Format expected: Timestamp, P1_ID, P1_Activity, P2_ID, P2_Activity
    """
    def __init__(self, csv_path, stale_threshold_seconds=3600):
        # Default to 3600s (1 hour) to handle sleeping subjects who don't move
        super().__init__(csv_path, stale_threshold_seconds=stale_threshold_seconds)
        self.latest_p1_activity = "NA"
        self.latest_p2_activity = "NA"
        self.latest_timestamp = None
        self.lock = threading.Lock()
        self.fall_detected = False
        self.last_fall_reason = "" # Tracks specific string to send to GUI
        
        # Buffer for inactivity detection (stores tuples: (timestamp, p1_raw, p2_raw))
        self.history_buffer = deque()
        
        self.activity_map = {
            "Walking": "standing_walking",
            "Transition-LayFloor-to-Stand": "standing_walking",
            "Transition-Sit-to-Stand": "standing_walking",
            "Falling": "Fall",
            "Transition-Stand-to-Sit": "sitting",
            "Transition-LayBed-to-Sit": "sitting",
            "Sit-Stationary": "sitting",
            "Transition-Sit-to-LayBed": "lying",
            "LayBed-Stationary": "lying",
            "LayFloor-Stationary": "lying",
            "NA": "NA"
        }
            
    def _parse_csv_line(self, line):
        try:
            parts = line.split(',')
            # Expecting at least 5 columns for 2 people: Timestamp, ID1, Act1, ID2, Act2
            if len(parts) < 5:
                while len(parts) < 5:
                    parts.append("NA")
            
            timestamp_str = parts[0].strip()
            # Person 1 is index 2, Person 2 is index 4 (0-based)
            raw_p1 = parts[2].strip()
            raw_p2 = parts[4].strip()
            
            try:
                timestamp = int(float(timestamp_str))
            except ValueError:
                return
            
            simplified_p1 = self.activity_map.get(raw_p1, raw_p1)
            simplified_p2 = self.activity_map.get(raw_p2, raw_p2)
            
            with self.lock:
                # Update buffers for inactivity detection
                self.history_buffer.append((timestamp, raw_p1, raw_p2))
                
                # Cleanup buffer: keep last 15 seconds to be safe for 10s window check
                current_time = time.time()
                while self.history_buffer and (current_time - self.history_buffer[0][0] > 15):
                    self.history_buffer.popleft()

                # Trigger fall logic immediately for Fall OR LayFloor-Stationary
                # Update last_fall_reason for the GUI display
                if simplified_p1 == "Fall" or simplified_p2 == "Fall":
                    self.fall_detected = True
                    self.last_fall_reason = "Sudden Collapse"
                elif raw_p1 == "LayFloor-Stationary" or raw_p2 == "LayFloor-Stationary":
                    self.fall_detected = True
                    self.last_fall_reason = "Man Down"
                
                self.latest_p1_activity = simplified_p1
                self.latest_p2_activity = simplified_p2
                self.latest_timestamp = timestamp
                logger.debug(f"TI Radar: P1='{simplified_p1}', P2='{simplified_p2}'")

        except Exception as e:
            logger.error(f"Error parsing TI CSV line '{line}': {e}")

    def check_recent_sedentary(self, window_seconds=10):
        """
        Checks if the last 'window_seconds' of history contains any of the specified
        sedentary keywords for EITHER person.
        """
        target_states = {
            "Transition-Stand-to-Sit", 
            "Transition-LayBed-to-Sit", 
            "Sit-Stationary", 
            "Transition-Sit-to-LayBed", 
            "LayBed-Stationary"
        }
        
        threshold_time = time.time() - window_seconds
        
        with self.lock:
            # Iterate through buffer. If ANY valid activity matches target states, return True.
            for ts, p1_act, p2_act in self.history_buffer:
                if ts >= threshold_time:
                    if p1_act in target_states or p2_act in target_states:
                        return True
        return False
            
    def get_latest_activities(self):
        """Get the latest activities and timestamp, handling the fall latch."""
        with self.lock:
            # Fall latch applies globally to the event, but we return individual states
            is_fall = self.fall_detected
            reason = self.last_fall_reason if is_fall else ""
            if self.fall_detected:
                self.fall_detected = False
            
            return self.latest_p1_activity, self.latest_p2_activity, self.latest_timestamp, is_fall, reason

# === NOVELDA RADAR CSV MONITOR ===
class NoveldaHealthMonitor(BaseRadarCsvMonitor):
    """Monitors the Novelda BPM CSV file purely for health checking."""
    def __init__(self, csv_path, stale_threshold_seconds=300):
        # Default to 300s (5 mins) since Novelda writes constantly
        super().__init__(csv_path, stale_threshold_seconds=stale_threshold_seconds)
        logger.debug(f"NoveldaHealthMonitor initialized for {csv_path}")

    def _parse_csv_line(self, line):
        pass 

# === ENHANCED MQTT CLIENT ===
class EnhancedMQTTClient:
    """MQTT client with connection monitoring and message buffering"""
    def __init__(self, config, message_buffer):
        self.config = config
        self.buffer = message_buffer
        self.client = mqtt.Client()
        self.connected = False
        self.connection_thread = None
        self.resend_thread = None
        self.periodic_resend_thread = None
        self.connection_lock = threading.Lock()
        self.message_handler = None
        self.setup_callbacks()
        self.start_periodic_buffer_check()
    
    def setup_callbacks(self):
        """Setup MQTT callbacks"""
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_message = self._on_message
    
    def _on_message(self, client, userdata, msg):
        """Callback for incoming messages"""
        if self.message_handler:
            try:
                self.message_handler(client, userdata, msg)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
        else:
            logger.debug(f"Received message on topic {msg.topic} but no handler is set.")

    def start_periodic_buffer_check(self):
        """Start periodic check for stuck buffered messages"""
        def periodic_check():
            while True:
                time.sleep(30)
                try:
                    with self.connection_lock:
                        is_connected = self.connected
                        buffered_count = len(self.buffer.get_messages())
                    
                    if is_connected and buffered_count > 0:
                        logger.debug(f"Periodic check found {buffered_count} buffered messages while connected")
                        self.start_resend_thread()
                except Exception as e:
                    logger.error(f"Error in periodic buffer check: {e}")
        
        self.periodic_resend_thread = threading.Thread(target=periodic_check)
        self.periodic_resend_thread.daemon = True
        self.periodic_resend_thread.start()
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        if rc == 0:
            logger.info("Connected to MQTT broker") # Changed to INFO for status visibility
            with self.connection_lock:
                self.connected = True
        else:
            logger.error(f"Failed to connect, return code {rc}")
            with self.connection_lock:
                self.connected = False
            self.start_connection_thread()
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for disconnection"""
        logger.warning(f"Disconnected from MQTT broker (rc={rc})")
        with self.connection_lock:
            self.connected = False
        
        if rc != 0:
            logger.info("Unexpected disconnection - starting reconnection attempts")
            self.start_connection_thread()
    
    def _on_publish(self, client, userdata, mid):
        """Callback for successful publish"""
        logger.debug(f"Message {mid} published successfully")
    
    def connect(self):
        """Connect to MQTT broker with TLS if configured"""
        try:
            username = self.config.get('MQTT', 'username')
            password = self.config.get('MQTT', 'password')
            
            if username and password:
                self.client.username_pw_set(username, password)
                logger.debug(f"MQTT credentials set for user: {username}")

            if self.config.get('MQTT', 'use_tls') == 'true':
                ca_cert_path = self.config.get('TLS', 'ca_cert')
                
                if not os.path.exists(ca_cert_path):
                    logger.error(f"Missing CA certificate file. Check config path: {ca_cert_path}")
                    return False

                self.client.tls_set(ca_certs=ca_cert_path)
            
            broker = self.config.get('MQTT', 'broker')
            port = int(self.config.get('MQTT', 'port'))
            logger.debug(f"Connecting to broker {broker}:{port}")
            self.client.connect_async(broker, port)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            with self.connection_lock:
                self.connected = False
            self.start_connection_thread()
            return False
    
    def start_connection_thread(self):
        """Start thread to handle reconnection"""
        if self.connection_thread is None or not self.connection_thread.is_alive():
            self.connection_thread = threading.Thread(target=self._reconnect_loop)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            logger.debug("Connection thread started")
    
    def _reconnect_loop(self):
        """Continuously try to reconnect"""
        backoff = 1
        max_backoff = 60
        
        while True:
            # First, check if Paho has already reconnected automatically
            if self.client.is_connected():
                with self.connection_lock:
                    self.connected = True
                logger.debug("Already connected (Paho handled it), exiting reconnect loop")
                return

            with self.connection_lock:
                if self.connected:
                    logger.debug("Already connected, exiting reconnect loop")
                    return
            
            try:
                logger.debug(f"Attempting to reconnect in {backoff}s...")
                time.sleep(backoff)
                
                # Double-check connectivity before forcing anything
                if self.client.is_connected():
                    with self.connection_lock:
                        self.connected = True
                    return

                broker = self.config.get('MQTT', 'broker')
                port = int(self.config.get('MQTT', 'port'))
                
                logger.debug(f"Reconnecting to {broker}:{port}")
                try:
                    self.client.connect_async(broker, port)
                except Exception as e:
                    logger.error(f"connect_async failed during reconnect: {e}")
                
                time.sleep(3)
                
                # Check if successful
                if self.client.is_connected():
                     with self.connection_lock:
                        self.connected = True
                     logger.info("Reconnection successful") # Changed to INFO for status visibility
                     return
                
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                logger.error(f"Reconnection loop error: {e}")
                backoff = min(backoff * 2, max_backoff)
    
    def start_resend_thread(self):
        """Start thread to resend buffered messages"""
        if self.resend_thread is None or not self.resend_thread.is_alive():
            logger.debug("Starting new resend thread")
            self.resend_thread = threading.Thread(target=self._resend_buffered_messages)
            self.resend_thread.daemon = True
            self.resend_thread.start()
    
    def _resend_buffered_messages(self):
        """Resend all buffered messages when connection is restored"""
        messages = self.buffer.get_messages()
        if not messages:
            return
        
        total_messages = len(messages)
        messages_to_remove_count = 0
        
        for i, msg in enumerate(messages):
            with self.connection_lock:
                is_connected = self.connected
            
            if not is_connected:
                break
            
            try:
                info = self.client.publish(
                    msg['topic'],
                    msg['payload'],
                    qos=msg['qos'],
                    retain=False
                )
                
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    messages_to_remove_count += 1
                elif info.rc in [mqtt.MQTT_ERR_PAYLOAD_SIZE, mqtt.MQTT_ERR_INVAL]:
                    messages_to_remove_count += 1
                    logger.error(f"Discarding invalid message {i+1}")
                else:
                    with self.connection_lock:
                        self.connected = False
                    self.start_connection_thread()
                    break
                    
            except Exception as e:
                logger.error(f"Exception while resending: {e}")
                break
            
            time.sleep(0.05)
        
        if messages_to_remove_count > 0:
            self.buffer.remove_messages(messages_to_remove_count)
            logger.debug(f"Successfully resent {messages_to_remove_count} offline messages.")
            
            if len(self.buffer.get_messages()) > 0 and self.is_connected():
                threading.Timer(5.0, self.start_resend_thread).start()
    
    def publish(self, topic, payload, qos=0):
        """Publish message with automatic buffering on failure"""
        with self.connection_lock:
            is_connected = self.connected
        
        if is_connected:
            try:
                info = self.client.publish(topic, payload, qos)
                if info.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
                else:
                    logger.warning(f"Publish returned non-success code: {info.rc}")
                    with self.connection_lock:
                        self.connected = False
            except Exception as e:
                logger.error(f"Publish failed (Exception): {e}")
                with self.connection_lock:
                    self.connected = False
        
        # Buffer the message if not connected or if publish failed
        self.buffer.add_message(topic, payload, qos)
        
        # Only start the thread if we aren't already trying to reconnect
        self.start_connection_thread()
        return False
    
    def disconnect(self):
        """Disconnect from broker"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except:
            pass
        with self.connection_lock:
            self.connected = False

    def is_connected(self):
        """Thread-safe method to check connection status"""
        with self.connection_lock:
            return self.connected

    def set_message_handler(self, handler):
        """Set the callback for incoming messages."""
        self.message_handler = handler


# === MAIN APPLICATION ===
class RadarMonitoringSystem:
    """Main application class"""
    STALE_ACTIVITY_SECONDS = 60

    def __init__(self, shared_data_dir, version):
        self.shared_data_dir = shared_data_dir
        self.version = version
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.versions_dir = os.path.abspath(os.path.join(self.base_dir, '..'))

        self.config = DeviceConfig(shared_data_dir=self.shared_data_dir)
        
        # Keep track of the current date to detect when it changes
        self.current_date_str = datetime.now().strftime('%Y-%m-%d')
        
        self.buffer = MessageBuffer(shared_data_dir=self.shared_data_dir)
        self.mqtt_client = EnhancedMQTTClient(self.config, self.buffer)
        
        self.ti_monitor = TIRadarProcessMonitor()
        self.ti_activity_reader = TIActivityReader(
            csv_path=self.config.get('PATHS', 'ti_output_log')
        )
        self.novelda_health_monitor = NoveldaHealthMonitor(
            csv_path=self.config.get('PATHS', 'novelda_bpm')
        )
        
        self.novelda_proc = None
        self.ti_proc = None
        
        # Store for both persons
        self.current_p1_activity = "NA"
        self.current_p2_activity = "NA"
        self.current_activity_timestamp = None
        self.current_location = "Unknown Location"
        self.last_publish_time = 0
        self.publish_interval = 15
        
        self.fall_event_active = False
        
        self.locations = ["Cell Bunk", "Toilet", "Cell Door", "Chair", "Hallway", 
                         ]
        self.previous_bpm = None
        
        self.latest_novelda_bpm = "NA"
        self.latest_novelda_distance = None
        self.last_novelda_read_time = 0
        self.novelda_read_lock = threading.Lock()
        
        self.start_time = time.time()
        
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()

        self.ota_lock = threading.Lock()
        self.shutting_down_for_update = False
        
        # UI Properties (For PyQt5 Visuals)
        self.gui_bpm = "NA"
        self.gui_presence = 0
        self.gui_inactivity = "Empty"
        self.gui_alert_active = False
        self.gui_alert_reason = "None"
        
    def check_for_date_rollover(self):
        """Checks if the date has changed and updates paths if so"""
        now_date = datetime.now().strftime('%Y-%m-%d')
        if now_date != self.current_date_str:
            logger.info(f"Date rollover detected: {self.current_date_str} -> {now_date}")
            self.current_date_str = now_date
            
            # Update the config file with new daily paths and ensure files exist
            self.config.update_paths()
            self.config.save_config()
            
            # Update the CSV readers to point to new files
            new_ti_path = self.config.get('PATHS', 'ti_output_log')
            new_novelda_path = self.config.get('PATHS', 'novelda_bpm')
            
            self.ti_activity_reader.update_path(new_ti_path)
            self.novelda_health_monitor.update_path(new_novelda_path)

    def process_bpm_value(self, bpm):
        """Process BPM value according to specified rules"""
        if bpm is None or bpm == "NA":
            return "NA"
        
        if not isinstance(bpm, (int, float)):
            logger.warning(f"process_bpm_value received non-numeric value: {bpm}")
            return "NA"
        
        bpm_rounded = round(bpm, 2)
        if bpm_rounded == 30.00:
            logger.debug("BPM is exactly 30.00, publishing 'NA' instead")
            return "NA"
        
        self.previous_bpm = bpm_rounded
        return bpm_rounded
       
    def provision_device(self):
        """Handle device provisioning"""
        logger.info("Using device configuration from file.")
        pass
    
    def launch_novelda(self):
        """Launch Novelda radar script"""
        try:
            script_name = "novelda.py"
            script_path = os.path.join(self.base_dir, script_name)

            if not os.path.exists(script_path):
                logger.error(f"Novelda script not found at {script_path}")
                return None

            self.novelda_proc = subprocess.Popen(
                [
                    sys.executable, 
                    script_name, 
                    "--shared_data_dir", self.shared_data_dir,
                    "--version", self.version 
                ],
                cwd=self.base_dir
            )
            logger.debug(f"Novelda radar process started from {self.base_dir}")
            return self.novelda_proc
        except Exception as e:
            logger.error(f"Failed to start Novelda: {e}")
            return None
    
    def launch_ti_radar(self):
        """Launch TI radar script with stdout capture"""
        try:
            script_name = "ti.py"
            script_path = os.path.join(self.base_dir, script_name)

            if not os.path.exists(script_path):
                logger.error(f"TI script not found at {script_path}")
                return None

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.ti_proc = subprocess.Popen(
                [
                    sys.executable, 
                    "-u", script_name, 
                    "--shared_data_dir", self.shared_data_dir,
                    "--version", self.version
                ],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=0,
                env=env
            )
            
            logger.debug(f"TI radar process started from {self.base_dir}")
            self.ti_monitor.start_reading(self.ti_proc)
            return self.ti_proc
        except Exception as e:
            logger.error(f"Failed to start TI radar: {e}")
            return None

    def _read_last_line(self, csv_path, data_name_for_log):
        """
        Helper to read the last non-empty row from a CSV.
        Includes retry logic for Windows file locking and efficient seeking.
        Returns (row_data_list, timestamp_int)
        """
        if not os.path.exists(csv_path):
            logger.debug(f"Novelda {data_name_for_log} CSV not found: {csv_path}")
            return None, None
        
        last_row_seen = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                with open(csv_path, 'rb') as f:
                    try:
                        f.seek(-1024, os.SEEK_END)
                    except OSError:
                        f.seek(0)
                    
                    lines = f.readlines()
                    decoded_lines = [line.decode('utf-8').strip() for line in lines if line.strip()]
                    
                    if not decoded_lines:
                        f.seek(0)
                        lines = f.readlines()
                        decoded_lines = [line.decode('utf-8').strip() for line in lines if line.strip()]

                    if not decoded_lines:
                         return None, None
                    
                    last_line = decoded_lines[-1]
                    
                    if "timestamp" in last_line.lower():
                        return None, None
                        
                    reader = csv.reader([last_line])
                    last_row_seen = next(reader)
                    
                    break
            except (PermissionError, IOError):
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error reading Novelda {data_name_for_log} CSV: {e}")
                return None, None
        
        if last_row_seen and len(last_row_seen) >= 1:
            try:
                timestamp = int(float(last_row_seen[0].strip()))
                return last_row_seen, timestamp
            except (ValueError, TypeError):
                logger.warning(f"{data_name_for_log} CSV: invalid timestamp: {last_row_seen[0]}")
                return last_row_seen, None
        
        return None, None

    def get_latest_novelda_rr(self):
        """
        Get the latest respiration rate from Novelda BPM CSV.
        """
        csv_path = self.config.get('PATHS', 'novelda_bpm')
        last_row, bpm_timestamp = self._read_last_line(csv_path, "BPM")

        bpm_to_process = "NA" 
        current_time = time.time()

        if bpm_timestamp is None or (current_time - bpm_timestamp > self.STALE_ACTIVITY_SECONDS):
            if bpm_timestamp is not None:
                logger.error(f"Novelda BPM CSV file is stale! Last update {current_time - bpm_timestamp:.0f}s ago. Sending 'NA'.")
            bpm_to_process = "NA"
        elif last_row:
            if len(last_row) >= 2:
                bpm_str = last_row[1].strip()
                if bpm_str == "NA":
                    bpm_to_process = "NA"
                else:
                    try:
                        bpm_to_process = float(bpm_str)
                    except (ValueError, TypeError):
                        bpm_to_process = "NA"
        
        self.latest_novelda_bpm = self.process_bpm_value(bpm_to_process)
        return self.latest_novelda_bpm

    def get_latest_novelda_distance(self):
        """
        Read the latest distance from the Novelda UWB CSV.
        """
        with self.novelda_read_lock:
            current_time = time.time()
            if (current_time - self.last_novelda_read_time < 1.0):
                return self.latest_novelda_distance
            
            csv_path = self.config.get('PATHS', 'novelda_dis')
            last_row, _ = self._read_last_line(csv_path, "Distance")

            new_distance = None 

            if last_row:
                if len(last_row) >= 2:
                    dist_str = last_row[1].strip()
                    if dist_str == "NA":
                        new_distance = None
                    else:
                        try:
                            new_distance = float(dist_str)
                        except (ValueError, TypeError):
                            new_distance = None
            
            self.latest_novelda_distance = new_distance
            self.last_novelda_read_time = current_time 
            return self.latest_novelda_distance

    def get_latest_ti_raw_distance(self):
        """Read the latest 'dist_uwb' from the TI raw data CSV."""
        try:
            csv_path = self.config.get('PATHS', 'ti_raw_log')
            last_row, _ = self._read_last_line(csv_path, "TI Raw")
            
            if last_row and len(last_row) > 8:
                try:
                    if len(last_row) >= 12:
                        dist_val = float(last_row[11])
                        return dist_val
                except:
                    pass
            return None
        except Exception as e:
            logger.error(f"Error reading TI raw distance: {e}")
            return None

    def get_latest_presence_count(self):
        """Get the latest person count from num_people.csv"""
        csv_path = self.config.get('PATHS', 'num_people')
        last_row, _ = self._read_last_line(csv_path, "Person Count")
        
        if last_row and len(last_row) >= 2:
            try:
                return int(last_row[1].strip())
            except (ValueError, TypeError):
                logger.warning(f"Invalid person count value in {csv_path}: {last_row[1]}")
                return 0 
        return 0 # Default if no file or empty

    def get_simulated_location(self):
        """Get simulated location"""
        return random.choice(self.locations)
    
    def get_latest_ti_activities(self):
        """Get latest TI radar activities from CSV, with UWB distance validation."""
        ti_distance = self.get_latest_ti_raw_distance()
        novelda_distance = self.get_latest_novelda_distance()

        is_valid_target = True
        log_status = "parsed"
        
        if ti_distance is None or novelda_distance is None:
            log_status = "unverified"
        else:
            distance_diff = abs(ti_distance - novelda_distance)
            if distance_diff > 1:
                is_valid_target = False
                log_status = "ghost"
                logger.debug(
                    f"Ghost detected. TI: {ti_distance:.2f}m, Novelda: {novelda_distance:.2f}m"
                )

        self.log_fusion_data(time.time(), ti_distance, novelda_distance, log_status)

        p1_act, p2_act, timestamp, is_fall_event, fall_reason = self.ti_activity_reader.get_latest_activities()

        if timestamp is not None:
             self.current_activity_timestamp = timestamp
        
        if not is_valid_target:
             return self.current_p1_activity, self.current_p2_activity, self.current_activity_timestamp, self.get_simulated_location(), False, ""
        
        if is_fall_event:
            return p1_act, p2_act, timestamp, self.get_simulated_location(), True, fall_reason
        else:
            self.current_p1_activity = p1_act
            self.current_p2_activity = p2_act
            return self.current_p1_activity, self.current_p2_activity, self.current_activity_timestamp, self.get_simulated_location(), False, ""
    
    def build_activity_topic(self):
        tenant_id = self.config.get('MQTT', 'tenant_id')
        device_id = self.config.get('MQTT', 'device_id')
        return f"cell/{tenant_id}/devices/{device_id}/health/activity"
    
    def build_fall_topic(self):
        tenant_id = self.config.get('MQTT', 'tenant_id')
        device_id = self.config.get('MQTT', 'device_id')
        return f"cell/{tenant_id}/devices/{device_id}/health/fall"

    def build_heartbeat_topic(self):
        tenant_id = self.config.get('MQTT', 'tenant_id')
        device_id = self.config.get('MQTT', 'device_id')
        return f"cell/{tenant_id}/devices/{device_id}/health/heartbeat"

    def build_ota_command_topic(self):
        return "cell/+/ota/announce"

    def build_ota_status_topic(self):
        tenant_id = self.config.get('MQTT', 'tenant_id')
        device_id = self.config.get('MQTT', 'device_id')
        return f"cell/{tenant_id}/devices/{device_id}/ota/status"

    def publish_activity(self, timestamp, location, p1_activity, p2_activity, bpm, presence_count, inactivity_status):
        payload = {
            "timestamp": int(timestamp),
            "device_id": self.config.get('MQTT', 'device_id'),
            "location": location,
            "person1_activity": p1_activity,
            "person2_activity": p2_activity,
            "bpm": bpm if bpm != "NA" else "NA",
            "presence_count": presence_count,
            "inactivity_status": inactivity_status
        }
        
        topic = self.build_activity_topic()
        success = self.mqtt_client.publish(topic, json.dumps(payload), qos=0)
        if success:
            logger.debug(f"Activity published: {payload}")
        return success
    
    def publish_fall(self, timestamp, location, presence_count, inactivity_status, p1_activity, p2_activity):
        payload = {
            "timestamp": int(timestamp),
            "device_id": self.config.get('MQTT', 'device_id'),
            "location": location,
            "person1_activity": p1_activity,
            "person2_activity": p2_activity,
            "presence_count": presence_count,
            "inactivity_status": inactivity_status
        }
        topic = self.build_fall_topic()
        success = self.mqtt_client.publish(topic, json.dumps(payload), qos=2)
        return success
    
    def publish_ota_status(self, status, new_version, message=None):
        topic = self.build_ota_status_topic()
        payload = {
            "timestamp": int(time.time()),
            "device_id": self.config.get('MQTT', 'device_id'),
            "current_version": self.version,
            "target_version": new_version,
            "status": status,
        }
        if message:
            payload["message"] = message
        logger.info(f"Publishing OTA Status: {status} - {message or ''}")
        self.mqtt_client.publish(topic, json.dumps(payload), qos=1)

    def get_system_stats(self):
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_path = self.shared_data_dir if os.path.exists(self.shared_data_dir) else '/'
            disk_usage = psutil.disk_usage(disk_path)
            disk_free_gb = round(disk_usage.free / (1024**3), 2)
            return cpu_percent, memory_percent, disk_free_gb
        except Exception:
            return None, None, None

    def save_heartbeat_to_csv(self, payload):
        log_path = self.config.get('PATHS', 'heartbeat_log')
        if not log_path:
            return

        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_exists = os.path.exists(log_path)
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(payload.keys())
                row_data = payload.copy()
                row_data['system_stats'] = json.dumps(row_data['system_stats'])
                row_data['last_recording_status'] = json.dumps(row_data['last_recording_status'])
                row_data['error_codes'] = json.dumps(row_data['error_codes'])
                writer.writerow(row_data.values())
        except Exception as e:
            logger.error(f"Failed to write heartbeat to CSV: {e}")

    def _heartbeat_loop(self):
        while not self.heartbeat_stop_event.wait(15.0):
            try:
                if self.shutting_down_for_update:
                    break
                
                self.check_for_date_rollover()

                ts_now = time.time()
                timestamp_int = int(ts_now)
                timestamp_str = f"{ts_now:.4f}"

                device_id = self.config.get('MQTT', 'device_id')
                uptime_seconds = int(ts_now - self.start_time)
                
                cpu, mem, disk_free = self.get_system_stats()
                
                novelda_status = "running" if self.novelda_health_monitor.is_healthy() else "offline"
                ti_status = "running" if self.ti_activity_reader.is_healthy() else "offline"
                
                payload = {
                    "timestamp": timestamp_int,
                    "device_id": device_id,
                    "uptime_seconds": uptime_seconds,
                    "version": self.version,
                    "system_stats": {
                        "cpu_percent": cpu,
                        "memory_percent": mem,
                        "disk_free_gb": disk_free
                    },
                    "last_recording_status": {
                        "novelda_label": novelda_status,
                        "ti_label": ti_status
                    },
                    "error_codes": []
                }
                
                csv_payload = payload.copy()
                csv_payload['timestamp'] = timestamp_str
                self.save_heartbeat_to_csv(csv_payload)
                
                self.mqtt_client.publish(self.build_heartbeat_topic(), json.dumps(payload), qos=0)
                logger.debug("Heartbeat sent.")
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    def start_heartbeat(self):
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_stop_event.clear()
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
            self.heartbeat_thread.daemon = True
            self.heartbeat_thread.start()

    def stop_heartbeat(self):
        try:
            self.heartbeat_stop_event.set()
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=2.0)
        except Exception:
            pass
            
    def log_to_unified(self, timestamp, location, p1_activity, p2_activity, bpm, presence_count, inactivity_status, is_fall=False):
        output_path = self.config.get('PATHS', 'output_path')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        entry = {
            "timestamp": int(timestamp),
            "device_id": self.config.get('MQTT', 'device_id'),
            "location": location,
            "person1_activity": p1_activity,
            "person2_activity": p2_activity,
            "bpm": bpm if (not is_fall or bpm is not None) else "NA",
            "presence_count": presence_count,
            "inactivity_status": inactivity_status,
            "is_fall": is_fall,
            "topic": self.build_fall_topic() if is_fall else self.build_activity_topic()
        }
        
        try:
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to unified log: {e}")
    
    def log_fusion_data(self, timestamp, ti_dist, novelda_dist, status):
        log_path = self.config.get('PATHS', 'fusion_log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        entry = {
            "timestamp": int(timestamp),
            "ti_dist_uwb": ti_dist if ti_dist is not None else "NA",
            "novelda_uwb_dist": novelda_dist if novelda_dist is not None else "NA",
            "status": status
        }
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

    def get_status_display(self):
        mqtt_status = "Connected" if self.mqtt_client.connected else "Offline"
        buffered_msgs = len(self.buffer.get_messages())
        buffer_status = f"{buffered_msgs} messages buffered" if buffered_msgs > 0 else "No buffered messages"
        
        novelda_status = "OK" if self.novelda_health_monitor.is_healthy() else "Check/Offline"
        ti_status = "OK" if self.ti_activity_reader.is_healthy() else "Check/Offline"
        
        return f"MQTT: {mqtt_status} | Novelda: {novelda_status} | TI: {ti_status} | Buffer: {buffer_status}"
    
    def monitor_and_publish(self):
        """Main monitoring loop"""
        logger.debug("Starting monitoring loop (publishing every 15s)...")
        
        self.last_publish_time = time.time()
        status_display_counter = 0
        
        while not self.shutting_down_for_update:
            try:
                self.check_for_date_rollover()

                current_time = time.time()
                
                status_display_counter += 1
                if status_display_counter >= 60:
                    logger.info(f"System Status: {self.get_status_display()}")
                    status_display_counter = 0
                
                p1_act, p2_act, activity_timestamp, location, is_fall_detected, fall_reason = self.get_latest_ti_activities()
                rr = self.get_latest_novelda_rr()
                is_low_bpm = isinstance(rr, (int, float)) and rr < 10

                # --- REAL-TIME CALCULATION FOR GUI ---
                presence = self.get_latest_presence_count()
                has_current_activity = (p1_act != "NA") or (p2_act != "NA")
                
                inactivity_status = "Active" 
                if presence == 0 and not has_current_activity and (rr == "NA" or rr is None):
                     inactivity_status = "Empty"
                elif presence >= 1:
                    if not has_current_activity:
                        has_valid_bpm = (rr != "NA" and rr is not None)
                        if has_valid_bpm or self.ti_activity_reader.check_recent_sedentary(window_seconds=10):
                            inactivity_status = "Inactive"
                        else:
                            inactivity_status = "Active"
                    else:
                        inactivity_status = "Active"
                elif presence == 0:
                    inactivity_status = "Empty"

                # Update variables for PyQt5 GUI display
                self.gui_bpm = rr if rr is not None else "NA"
                self.gui_presence = presence
                self.gui_inactivity = inactivity_status
                # -------------------------------------

                
                if is_fall_detected or is_low_bpm:
                    # Capture exact Alert Reason for GUI
                    self.gui_alert_active = True
                    if is_fall_detected:
                        self.gui_alert_reason = fall_reason
                    else:
                        self.gui_alert_reason = "Low_BPM"

                    if not self.fall_event_active:
                        if is_fall_detected:
                            logger.critical("IMMEDIATE FALL DETECTED - PUBLISHING")
                            bpm_to_log = None
                        else:
                            logger.critical(f"IMMEDIATE LOW BPM DETECTED ({rr}) - PUBLISHING")
                            p1_act = "Low_BPM"
                            p2_act = "Low_BPM"
                            bpm_to_log = rr
                        
                        # Override logic for emergency states
                        inactivity_status = "Active"
                        self.gui_inactivity = "Active" 

                        self.log_to_unified(current_time, location, p1_act, p2_act, bpm_to_log, presence, inactivity_status, is_fall=True)
                        success = self.publish_fall(current_time, location, presence, inactivity_status, p1_act, p2_act)
                        
                        self.fall_event_active = True
                        self.last_publish_time = current_time
                else: 
                    # Drop GUI alert flag when safe
                    self.gui_alert_active = False
                    self.gui_alert_reason = "None"

                    if self.fall_event_active:
                        self.fall_event_active = False
                    
                    if activity_timestamp is None or (current_time - activity_timestamp > self.STALE_ACTIVITY_SECONDS):
                        p1_act = "NA"
                        p2_act = "NA"

                    if current_time - self.last_publish_time >= self.publish_interval:
                        logger.debug(f"Read BPM from CSV: {rr}") 

                        self.log_to_unified(current_time, location, p1_act, p2_act, rr, presence, inactivity_status, is_fall=False)
                        success = self.publish_activity(current_time, location, p1_act, p2_act, rr, presence, inactivity_status)
                    
                        self.last_publish_time = current_time
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1)
        
        logger.info("Monitoring loop has exited.")
    
    def send_health_ping(self):
        try:
            ping_file = os.path.join(self.shared_data_dir, "health.ping")
            with open(ping_file, 'w') as f:
                f.write(str(int(time.time())))
        except Exception:
            pass
    
    def check_for_rollback_log(self):
        log_file = os.path.join(self.shared_data_dir, "rollback.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    reason = f.read().strip()
                logger.warning(f"Rollback detected! Reason: {reason}")
                time.sleep(2) 
                self.publish_ota_status("error", "unknown", f"Rollback detected: {reason}")
                os.remove(log_file)
            except Exception:
                pass

    def subscribe_to_ota_topic(self):
        try:
            topic = self.build_ota_command_topic()
            self.mqtt_client.client.subscribe(topic, qos=1)
            self.mqtt_client.set_message_handler(self._on_mqtt_message)
        except Exception as e:
            logger.error(f"Failed to subscribe to OTA topic: {e}")

    def _on_mqtt_message(self, client, userdata, msg):
        command_topic_sub = self.build_ota_command_topic()
        if mqtt.topic_matches_sub(command_topic_sub, msg.topic):
            logger.info(f"OTA command received on matching topic: {msg.topic}")
            threading.Thread(target=self._handle_ota_command, args=(msg.payload,)).start()

    def _handle_ota_command(self, payload_bytes):
        with self.ota_lock:
            if self.shutting_down_for_update:
                return
            
            logger.info("--- Starting OTA Update Process ---")

            try:
                data = json.loads(payload_bytes.decode('utf-8'))
                target_device_id = data.get('device_id')
                my_device_id = self.config.get('MQTT', 'device_id')
                if target_device_id and target_device_id != my_device_id:
                     logger.debug(f"OTA ignored: target {target_device_id} != my id {my_device_id}")
                     return

                update_url = data.get('url')
                new_version = data.get('version')
                checksum = data.get('checksum')

                if not update_url or not new_version:
                     logger.error("OTA payload missing url or version")
                     self.publish_ota_status("error", "unknown", "Missing URL/version")
                     return
                
                if new_version == self.version:
                     logger.info(f"Already on version {new_version}. Ignoring.")
                     self.publish_ota_status("success", new_version, "Already up to date")
                     return

                logger.info(f"Downloading update v{new_version} from {update_url}")
                self.publish_ota_status("downloading", new_version)

                zip_path = os.path.join(self.shared_data_dir, "update_package.zip")
                try:
                    r = requests.get(update_url, stream=True, timeout=30)
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    logger.error(f"Download failed: {e}")
                    self.publish_ota_status("error", new_version, f"Download failed: {e}")
                    return

                if checksum:
                    hasher = hashlib.sha256()
                    with open(zip_path, 'rb') as f:
                        while True:
                            chunk = f.read(8192)
                            if not chunk:
                                break
                            hasher.update(chunk)
                    calculated_hash = hasher.hexdigest()
                    if calculated_hash.lower() != checksum.lower():
                        logger.error("Checksum mismatch!")
                        self.publish_ota_status("error", new_version, "Checksum mismatch")
                        return

                logger.info("Download verified. Triggering update installation...")
                self.publish_ota_status("installing", new_version)
                
                trigger_file = os.path.join(self.shared_data_dir, "trigger_update.json")
                with open(trigger_file, 'w') as f:
                    json.dump({"version": new_version, "zip_path": zip_path}, f)

                self.shutting_down_for_update = True
                logger.info("Exiting application to allow update...")
                
                self.stop_heartbeat()
                if self.novelda_proc: self.novelda_proc.terminate()
                if self.ti_proc: self.ti_proc.terminate()
                self.mqtt_client.disconnect()
                
                time.sleep(1)
                sys.exit(0)

            except Exception as e:
                logger.error(f"OTA Error: {e}")
                self.publish_ota_status("error", "unknown", f"Exception: {e}")

    def run(self):
        """Main execution"""
        try:
            logger.info(f"Starting Radar Monitoring System v{self.version}")
            self.config.generate_qr_code()
            self.check_for_rollback_log()
            
            self.launch_novelda()
            self.launch_ti_radar()
            self.ti_activity_reader.start_monitoring()
            self.novelda_health_monitor.start_monitoring()
            
            if self.mqtt_client.connect():
                self.subscribe_to_ota_topic()
            
            self.start_heartbeat()
            self.monitor_and_publish()
            
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down...")
            self.stop_heartbeat()
            with self.ota_lock:
                for proc in [self.novelda_proc, self.ti_proc]:
                    if proc and proc.poll() is None:
                        proc.terminate()
                
                self.mqtt_client.disconnect()
                logger.debug("Shutdown complete")


# === MINIMAL GUI ===
class MinimalDisplay(QWidget):
    """A compact, minimalist PyQt5 GUI that polls the app variables"""
    def __init__(self, radar_app):
        super().__init__()
        self.radar_app = radar_app
        self.alert_clear_time = 0  # Timestamp to hold the visual alert
        self.init_ui()
        
        # Setup timer to update the display twice a second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(500) 
        
    def init_ui(self):
        self.setWindowTitle("InvisiGuard Status")
        # Widened to 800 to fit 4 panels side-by-side
        self.resize(1000, 150)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Create the 4 Side-by-Side panels
        self.bpm_frame, self.bpm_label = self.create_panel("BPM", "NA")
        self.presence_frame, self.presence_label = self.create_panel("Presence", "0")
        self.status_frame, self.status_label = self.create_panel("Status", "Empty")
        self.alert_frame, self.alert_label = self.create_panel("Alert", "None")
        
        layout.addWidget(self.bpm_frame)
        layout.addWidget(self.presence_frame)
        layout.addWidget(self.status_frame)
        layout.addWidget(self.alert_frame)
        
        self.setLayout(layout)
        
    def create_panel(self, title, default_val):
        frame = QFrame()
        frame.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; color: white;")
        vbox = QVBoxLayout()
        vbox.setContentsMargins(10, 20, 10, 20)
        
        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setFont(QFont("Arial", 14))
        title_lbl.setStyleSheet("color: #aaaaaa; background: transparent;")
        
        val_lbl = QLabel(default_val)
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setFont(QFont("Arial", 24, QFont.Bold)) # Scaled font slightly for longer strings
        val_lbl.setStyleSheet("background: transparent;")
        
        vbox.addWidget(title_lbl)
        vbox.addWidget(val_lbl)
        frame.setLayout(vbox)
        
        return frame, val_lbl
        
    def update_display(self):
        """Polls the variables exposed by RadarMonitoringSystem"""
        self.bpm_label.setText(str(self.radar_app.gui_bpm))
        self.presence_label.setText(str(self.radar_app.gui_presence))
        self.status_label.setText(str(self.radar_app.gui_inactivity))
        
        # Update Alert Panel based on backend flag
        if self.radar_app.gui_alert_active:
            # Turn Red and display reason
            self.alert_frame.setStyleSheet("background-color: #d9534f; border-radius: 10px; color: white;")
            self.alert_label.setText(str(self.radar_app.gui_alert_reason))
            # Push the clear time 3 seconds into the future so it doesn't blink out instantly
            self.alert_clear_time = time.time() + 3.0 
        else:
            # Only turn back to Grey if the 3-second hold time has passed
            if time.time() > self.alert_clear_time:
                self.alert_frame.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; color: white;")
                self.alert_label.setText("None")
        
    def closeEvent(self, event):
        """Handle user closing the window natively"""
        logger.info("GUI window closed. Triggering shutdown...")
        self.radar_app.shutting_down_for_update = True
        event.accept()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InvisiGuard Radar Monitoring System")
    parser.add_argument(
        "--shared_data_dir",
        type=str,
        required=True,
        help="The absolute path to the shared_data directory."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="The current application version string."
    )
    args = parser.parse_args()
 
    log_file_path = os.path.join(args.shared_data_dir, "logs", "main_app.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)

    # Instantiate Core Application
    app = RadarMonitoringSystem(args.shared_data_dir, args.version)
    
    # Run the core logic in a daemon thread so it doesn't block the PyQt5 GUI
    system_thread = threading.Thread(target=app.run)
    system_thread.daemon = True
    system_thread.start()
    
    # Start the GUI in the main thread
    qt_app = QApplication(sys.argv)
    gui = MinimalDisplay(app)
    gui.show()
    
    # Block the main thread via the Qt Event Loop until window is closed
    sys.exit(qt_app.exec_())