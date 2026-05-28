#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated TI Radar GUI System
- Combines visualization/tracking from GUI_multi_person.py
- Combines argument parsing, path handling, and data logging from ti.py
- Ensures CSV file naming and timestamp formats match ti.py
- Conditional Logging: Only saves model output if subjects are detected.
"""

# === Standard libs ===
import os
import sys
import time
import csv
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# === Third-party ===
import numpy as np
import pandas as pd
import serial
import torch
from scipy.spatial.distance import cdist

# === GUI Imports ===
from PyQt5 import QtWidgets, QtCore
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

# ======================================================================================
# Configuration
# ======================================================================================

# Serial ports (Windows) - From ti.py
CLI_PORT = 'COM16'
DATA_PORT = 'COM15'

# Inference windowing
WINDOW_SIZE = 40
HOP_SIZE    = 10
FRAME_PAD_TARGET = 150  # points per frame fed to the model

# Multi-person thresholds - From GUI_multi_person.py
MIN_POINTS_PER_PERSON = 30
MAX_TRACKERS = 2
CLUSTER_EPS = 0.8
TRACK_MATCH_THRESH = 1.5
MAX_MISSING_FRAMES = 40

# ======================================================================================
# Tracking Logic (From GUI_multi_person.py)
# ======================================================================================

class Track:
    def __init__(self, track_id, centroid):
        self.id = track_id
        self.centroid = centroid
        self.missing_frames = 0
        self.buffer = [] # Sliding window buffer for this person
        self.prediction = "NA"
        self.points_history = []

    def update(self, new_centroid, new_points):
        self.centroid = new_centroid
        self.missing_frames = 0
        self.points_history = new_points

class RadarTracker:
    def __init__(self):
        self.tracks = {}  # id -> Track
        self.next_id = 1  # Start IDs at 1

    def update(self, points: np.ndarray):
        """
        points: (N, 4) array [x, y, z, doppler]
        Returns: dict of {track_id: points_array}
        """
        # 1. Clustering
        clusters = self._cluster_points(points)

        cluster_centroids = []
        cluster_points_list = []
        for c_pts in clusters:
            centroid = c_pts[:, :3].mean(axis=0)
            cluster_centroids.append(centroid)
            cluster_points_list.append(c_pts)

        # 2. Association
        active_track_ids = list(self.tracks.keys())
        assignments = {}
        used_clusters = set()

        if active_track_ids and cluster_centroids:
            track_centroids = np.array([self.tracks[tid].centroid for tid in active_track_ids])
            dists = cdist(track_centroids, np.array(cluster_centroids))
            
            while True:
                if dists.size == 0: break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                val = dists[min_idx]
                
                if val > TRACK_MATCH_THRESH:
                    break 
                
                t_idx, c_idx = min_idx
                tid = active_track_ids[t_idx]
                
                if tid not in assignments and c_idx not in used_clusters:
                    assignments[tid] = c_idx
                    used_clusters.add(c_idx)
                    dists[t_idx, :] = np.inf
                    dists[:, c_idx] = np.inf
                else:
                    dists[t_idx, c_idx] = np.inf

        # 3. Update Tracks
        result_map = {}
        
        # A. Update matched
        for tid, c_idx in assignments.items():
            self.tracks[tid].update(cluster_centroids[c_idx], cluster_points_list[c_idx])
            result_map[tid] = cluster_points_list[c_idx]

        # B. New Tracks
        for i in range(len(cluster_centroids)):
            if i not in used_clusters:
                if len(self.tracks) < MAX_TRACKERS:
                    new_id = self.next_id
                    if 1 not in self.tracks: new_id = 1
                    elif 2 not in self.tracks: new_id = 2
                    else: 
                        new_id = self.next_id
                        self.next_id += 1
                    
                    new_track = Track(new_id, cluster_centroids[i])
                    new_track.points_history = cluster_points_list[i]
                    self.tracks[new_id] = new_track
                    result_map[new_id] = cluster_points_list[i]

        # C. Missing
        for tid in active_track_ids:
            if tid not in assignments:
                self.tracks[tid].missing_frames += 1

        # D. Prune
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.missing_frames > MAX_MISSING_FRAMES]
        for tid in dead_ids:
            del self.tracks[tid]
            
        return result_map

    def _cluster_points(self, points):
        if len(points) < MIN_POINTS_PER_PERSON:
            return []

        xyz = points[:, :3]
        n = len(xyz)
        visited = np.zeros(n, dtype=bool)
        clusters = []
        dmat = cdist(xyz, xyz)

        for i in range(n):
            if visited[i]: continue
            visited[i] = True
            cluster_indices = [i]
            queue = [i]
            while queue:
                curr = queue.pop(0)
                neighbors = np.where((dmat[curr] < CLUSTER_EPS) & (~visited))[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    cluster_indices.append(neighbor)
            
            if len(cluster_indices) >= MIN_POINTS_PER_PERSON:
                clusters.append(points[cluster_indices])
        
        clusters.sort(key=lambda x: len(x), reverse=True)
        return clusters

# ======================================================================================
# Serial & Parsing (From ti.py)
# ======================================================================================

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15, dtype='uint8')
byteBufferLength = 0

def serialConfig(cfg_path: str) -> Tuple[serial.Serial, serial.Serial]:
    global CLIport, Dataport
    CLIport = serial.Serial(CLI_PORT, 115200)
    Dataport = serial.Serial(DATA_PORT, 921600)

    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.rstrip('\r\n')
            CLIport.write((line + '\n').encode())
            time.sleep(0.01)
    return CLIport, Dataport

def parseConfigFile(cfg_path: str) -> Dict[str, float]:
    configParameters: Dict[str, float] = {}
    with open(cfg_path, 'r') as f:
        for line in f:
            splitWords = line.strip().split(" ")
            numRxAnt = 4
            numTxAnt = 3

            if "profileCfg" in splitWords[0]:
                startFreq = int(float(splitWords[2]))
                idleTime = int(splitWords[3])
                rampEndTime = float(splitWords[5])
                freqSlopeConst = float(splitWords[8])
                numAdcSamples = int(splitWords[10])
                numAdcSamplesRoundTo2 = 1
                while numAdcSamples > numAdcSamplesRoundTo2:
                    numAdcSamplesRoundTo2 *= 2
                digOutSampleRate = int(splitWords[11])

            elif "frameCfg" in splitWords[0]:
                chirpStartIdx = int(splitWords[1])
                chirpEndIdx = int(splitWords[2])
                numLoops = int(splitWords[3])
                numFrames = int(splitWords[4])

    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    return configParameters

def readAndParseData14xx(Dataport: serial.Serial, configParameters: Dict[str, float]):
    global byteBuffer, byteBufferLength

    MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS = 1020
    MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX = 1011
    
    maxBufferSize = 2**15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    magicOK = 0
    dataOK = 0
    frameNumber = 0
    detObj: Dict[str, object] = {}
    timestamp = 0.0

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec
        byteBufferLength += byteCount

    if byteBufferLength > 16:
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]
        startIdx = []
        for loc in possibleLocs:
            if np.all(byteBuffer[loc:loc+8] == magicWord):
                startIdx.append(loc)

        if startIdx:
            if 0 < startIdx[0] < byteBufferLength:
                shift = startIdx[0]
                byteBuffer[:byteBufferLength-shift] = byteBuffer[shift:byteBufferLength]
                byteBuffer[byteBufferLength-shift:] = np.zeros(len(byteBuffer[byteBufferLength-shift:]), dtype='uint8')
                byteBufferLength -= shift
            if byteBufferLength < 0: byteBufferLength = 0

            word = [1, 2**8, 2**16, 2**24]
            totalPacketLen = int(np.matmul(byteBuffer[12:16], word))
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    if magicOK:
        timestamp = time.time()
        word = [1, 2**8, 2**16, 2**24]
        idX = 0

        _magic = byteBuffer[idX:idX+8]; idX += 8
        _version = format(int(np.matmul(byteBuffer[idX:idX+4], word)), 'x'); idX += 4
        totalPacketLen = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4
        _platform = format(int(np.matmul(byteBuffer[idX:idX+4], word)), 'x'); idX += 4
        frameNumber = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4
        _timeCpuCycles = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4
        _numDetectedObj = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4
        numTLVs = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4
        _subFrameNumber = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4

        for _ in range(numTLVs):
            tlv_type = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4
            tlv_length = int(np.matmul(byteBuffer[idX:idX+4], word)); idX += 4

            if tlv_type == MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS:
                pointCloud: List[List[float]] = []
                elevationUnit = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]; idX += 4
                azimuthUnit   = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]; idX += 4
                dopplerUnit   = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]; idX += 4
                rangeUnit     = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]; idX += 4
                snrUnit       = np.frombuffer(byteBuffer[idX:idX+4], dtype=np.float32)[0]; idX += 4

                numPoints = (tlv_length - 20) // 8
                for _p in range(numPoints):
                    elevation = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.int8)[0]   * elevationUnit; idX += 1
                    azimuth   = np.frombuffer(byteBuffer[idX:idX+1], dtype=np.int8)[0]   * azimuthUnit;   idX += 1
                    doppler   = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.int16)[0]  * dopplerUnit;   idX += 2
                    rangeVal  = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.uint16)[0] * rangeUnit;     idX += 2
                    snr       = np.frombuffer(byteBuffer[idX:idX+2], dtype=np.uint16)[0] * snrUnit;       idX += 2

                    x = rangeVal * np.cos(elevation) * np.sin(azimuth)
                    y = rangeVal * np.cos(elevation) * np.cos(azimuth)
                    z = rangeVal * np.sin(elevation)
                    pointCloud.append([x, y, z, doppler, snr, elevation, azimuth, rangeVal])

                detObj['pointCloud'] = pointCloud
                dataOK = 1
            
            elif tlv_type == MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX:
                numBytes = tlv_length
                targetIndices = list(np.frombuffer(byteBuffer[idX:idX+numBytes], dtype=np.uint8))
                idX += numBytes
                detObj['targetIndices'] = targetIndices
            
            else:
                idX += tlv_length

        if idX > 0 and byteBufferLength >= totalPacketLen:
            shiftSize = totalPacketLen
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]), dtype='uint8')
            byteBufferLength -= shiftSize
            if byteBufferLength < 0: byteBufferLength = 0

    return dataOK, frameNumber, detObj, timestamp

# ======================================================================================
# Utilities
# ======================================================================================

def num_to_class(idx: int) -> str:
    mapping = {
        0: "Walking", 1: "Falling", 2: "Transition-LayFloor-to-Stand",
        3: "Transition-Stand-to-Sit", 4: "Transition-Sit-to-LayBed",
        5: "Transition-LayBed-to-Sit", 6: "Transition-Sit-to-Stand",
        7: "Sit-Stationary", 8: "LayBed-Stationary", 9: "LayFloor-Stationary",
    }
    return mapping.get(idx, "Unknown")

def fill_frame(points: List[List[float]], target_len: int) -> List[List[float]]:
    n = len(points)
    if n == 0: return [[0.0]*4] * target_len
    if n >= target_len: return points[:target_len]
    out = points.copy()
    for _ in range(target_len - n):
        j = random.randint(0, n - 1)
        out.append(points[j])
    return out

def append_row_csv(filepath: str, row: List[str]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='a', newline='') as f:
        csv.writer(f).writerow(row)

def save_df_csv(df: pd.DataFrame, filepath: str) -> None:
    if not df.empty:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, mode='a', index=False, header=not os.path.isfile(filepath))

def calculate_dist_angle_from_uwb(row):
    """From ti.py: Z is Height, Y is Depth, X is Width"""
    x_ti, y_ti, z_ti = row['x'], row['y'], row['z']
    
    x_uwb = x_ti
    y_uwb = y_ti 
    z_uwb = z_ti + 0.02 # Add vertical offset

    distance = np.sqrt(x_uwb**2 + y_uwb**2 + z_uwb**2)
    if distance == 0:
        return 0.0, 0.0, 0.0

    azimuth_rad = np.arctan2(x_uwb, y_uwb)   
    elevation_rad = np.arcsin(z_uwb / distance) 

    return distance, np.rad2deg(azimuth_rad), np.rad2deg(elevation_rad)

# ======================================================================================
# GUI Class
# ======================================================================================

class RadarGUI(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recording = False
        self.bounding_boxes: Dict[int, List[gl.GLLinePlotItem]] = {}

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        self.view = gl.GLViewWidget()
        vbox.addWidget(self.view)
        self.view.setCameraPosition(distance=15, elevation=30, azimuth=45)
        self.view.opts['center'] = QtGui.QVector3D(0, 0, 0)
        
        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)

        for line in self._create_cube(width=6, height=8, depth=3, y_translation=4):
            self.view.addItem(line)
        self._create_grid(cube_width=5, cube_height=3, cube_depth=5, grid_width=10, grid_height=10, spacing=0.5, cube_y_translation=0)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        self.btn = QtWidgets.QPushButton("Start Detecting")
        self.btn.setFixedSize(200, 250)
        self.btn.setStyleSheet("QPushButton { font-size: 16pt; }")
        self.btn.clicked.connect(self._toggle_recording)
        hbox.addWidget(self.btn)

        # Person 1 Panel
        vbox_p1 = QtWidgets.QVBoxLayout()
        self.label_title1 = QtWidgets.QLabel("Target 1")
        self.label_title1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_activity = QtWidgets.QLabel("Monitoring...")
        self.label_activity.setAlignment(QtCore.Qt.AlignCenter)
        self.label_activity.setFixedSize(300, 220)
        self.label_activity.setStyleSheet("QLabel { background-color: green; border: 1px solid black; font-size: 25pt; }")
        vbox_p1.addWidget(self.label_title1)
        vbox_p1.addWidget(self.label_activity)
        hbox.addLayout(vbox_p1)

        # Person 2 Panel
        vbox_p2 = QtWidgets.QVBoxLayout()
        self.label_title2 = QtWidgets.QLabel("Target 2")
        self.label_title2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_activity2 = QtWidgets.QLabel("Empty")
        self.label_activity2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_activity2.setFixedSize(300, 220)
        self.label_activity2.setStyleSheet("QLabel { background-color: gray; border: 1px solid black; font-size: 25pt; }")
        vbox_p2.addWidget(self.label_title2)
        vbox_p2.addWidget(self.label_activity2)
        hbox.addLayout(vbox_p2)

        # Quality Panel
        self.label_points = QtWidgets.QLabel("Points: 0")
        self.label_points.setAlignment(QtCore.Qt.AlignCenter)
        self.label_points.setFixedSize(200, 250)
        self.label_points.setStyleSheet("QLabel { background-color: gray; border: 1px solid black; font-size: 18pt; }")
        hbox.addWidget(self.label_points)
        hbox.addStretch()
        vbox.addLayout(hbox)

    def _toggle_recording(self):
        self.is_recording = not self.is_recording
        self.btn.setText("Stop Detecting" if self.is_recording else "Start Detecting")

    def _create_grid(self, cube_width, cube_height, cube_depth, grid_width, grid_height, spacing, cube_y_translation):
        z = cube_y_translation - (cube_height / 2)
        gx0 = 0 - (grid_width / 2)
        gy0 = 2.5 - (grid_height / 2)
        for y in np.arange(gy0, gy0 + grid_height + spacing, spacing):
            self.view.addItem(gl.GLLinePlotItem(pos=np.array([[gx0, y, z], [gx0 + grid_width, y, z]]), color=(0.5,0.5,0.5,1), width=1, antialias=True))
        for x in np.arange(gx0, gx0 + grid_width + spacing, spacing):
            self.view.addItem(gl.GLLinePlotItem(pos=np.array([[x, gy0, z], [x, gy0 + grid_height, z]]), color=(0.5,0.5,0.5,1), width=1, antialias=True))

    def _create_cube(self, width, height, depth, y_translation=0):
        v = np.array([
            [ width/2,  height/2 + y_translation,  depth/2], [ width/2, -height/2 + y_translation,  depth/2],
            [-width/2, -height/2 + y_translation,  depth/2], [-width/2,  height/2 + y_translation,  depth/2],
            [ width/2,  height/2 + y_translation, -depth/2], [ width/2, -height/2 + y_translation, -depth/2],
            [-width/2, -height/2 + y_translation, -depth/2], [-width/2,  height/2 + y_translation, -depth/2],
        ])
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        return [gl.GLLinePlotItem(pos=np.array([v[e[0]], v[e[1]]], dtype=np.float32), color=(1,0,0,1), width=2, antialias=True) for e in edges]

    def update_point_count_quality(self, n: int):
        self.label_points.setText(f"Points: {n}")
        if n > 150: color = "green"
        elif 50 <= n <= 150: color = "orange"
        else: color = "red"
        self.label_points.setStyleSheet(f"QLabel {{ background-color: {color}; border: 1px solid black; font-size: 18pt; }}")

    def update_activity(self, text: str, is_person1: bool):
        lbl = self.label_activity if is_person1 else self.label_activity2
        if text in ["Empty", "No Subject"]:
             lbl.setText(text)
             lbl.setStyleSheet("QLabel { background-color: gray; font-size: 18pt; }")
             return
        # Translate the raw text for the UI
        display_text = text
        if text == "Falling" or text == "LayFloor-Stationary":
            display_text = "Man Down"
        elif text == "LayBed-Stationary":
            display_text = "In Bunk"
        # Add any other translations you need here...

        if "Man Down" in display_text:
            lbl.setText(f"{display_text}")
            lbl.setStyleSheet("QLabel { background-color: red; font-size: 18pt; }")
        else:
            lbl.setText(display_text)
            lbl.setStyleSheet("QLabel { background-color: green; font-size: 18pt; }")

    def update_scatter_with_ids(self, points: np.ndarray, ids: List[int]):
        if len(points) == 0:
            self.scatter.setData(pos=np.zeros((0,3)))
            return
            
        colors = []
        for tid in ids:
            if tid == 1: colors.append((0.0, 1.0, 0.0, 1.0)) # Green P1
            elif tid == 2: colors.append((1.0, 1.0, 0.0, 1.0)) # Yellow P2
            elif tid == -1: colors.append((1.0, 0.0, 0.0, 0.5)) # Noise Red
            else: colors.append((0.5, 0.5, 1.0, 1.0)) # Other
        
        self.scatter.setData(pos=points, color=np.array(colors, dtype=np.float32), size=5.0)

    def draw_track_boxes(self, tracker):
        for tid in list(self.bounding_boxes.keys()):
            for item in self.bounding_boxes[tid]: self.view.removeItem(item)
        self.bounding_boxes.clear()

        for tid, track in tracker.tracks.items():
            if track.missing_frames > 5: continue
            
            c = track.centroid
            half = 0.5
            corners = np.array([
                [c[0]-half, c[1]-half, c[2]-half], [c[0]+half, c[1]-half, c[2]-half],
                [c[0]+half, c[1]+half, c[2]-half], [c[0]-half, c[1]+half, c[2]-half],
                [c[0]-half, c[1]-half, c[2]+half], [c[0]+half, c[1]-half, c[2]+half],
                [c[0]+half, c[1]+half, c[2]+half], [c[0]-half, c[1]+half, c[2]+half],
            ])
            edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            
            color = (0, 1, 0, 1) if tid == 1 else (1, 1, 0, 1)
            items = []
            for e in edges:
                line = gl.GLLinePlotItem(pos=np.array([corners[e[0]], corners[e[1]]]), color=color, width=2, antialias=True)
                self.view.addItem(line)
                items.append(line)
            self.bounding_boxes[tid] = items

# ======================================================================================
# Main
# ======================================================================================

def main(shared_data_dir: str):
    # --- Paths Setup (from ti.py) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configFileName = os.path.join(current_dir, '20241123_BPM_R5_edit.cfg')
    MODEL_PATH = os.path.join(current_dir, 'Pointnet_LSTM_HAR_model_2_epoch500.pt')
    
    PC_LOG_DIR = os.path.join(shared_data_dir, 'logs', 'TI_DATA')
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Global state for filenames to allow update on rollover
    state = {
        'POINTCLOUD_CSV': os.path.join(PC_LOG_DIR, f'radar_raw_log_{current_date_str}.csv'),
        'MODEL_OUT_CSV': os.path.join(PC_LOG_DIR, f'radar_output_log_{current_date_str}.csv'),
        'current_date_str': current_date_str
    }

    # --- Setup Serial ---
    try:
        if not os.path.exists(configFileName):
            print(f"Error: Config file not found at {configFileName}")
            sys.exit(1)
        CLI, DATA = serialConfig(configFileName)
        cfg = parseConfigFile(configFileName)
    except Exception as e:
        print(f"Error connecting to Radar: {e}")
        sys.exit(1)

    # --- Setup GUI & Model ---
    app = QtWidgets.QApplication([])
    gui = RadarGUI()
    gui.setWindowTitle('TI Radar - Multi-Person GUI')
    gui.resize(1280, 900)
    gui.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        model = model.to(device)
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # --- Init Tracker ---
    tracker = RadarTracker()

    # --- Header Check ---
    if not os.path.isfile(state['MODEL_OUT_CSV']):
        append_row_csv(state['MODEL_OUT_CSV'], ['timestamp', 'p1_id', 'p1_activity', 'p2_id', 'p2_activity'])

    # --- Main Loop (QTimer) ---
    def on_tick():
        try:
            # 1. Date Rollover Check (from ti.py)
            new_date_str = datetime.now().strftime('%Y-%m-%d')
            if new_date_str != state['current_date_str']:
                state['current_date_str'] = new_date_str
                state['POINTCLOUD_CSV'] = os.path.join(PC_LOG_DIR, f'radar_raw_log_{new_date_str}.csv')
                state['MODEL_OUT_CSV'] = os.path.join(PC_LOG_DIR, f'radar_output_log_{new_date_str}.csv')
                if not os.path.isfile(state['MODEL_OUT_CSV']):
                    append_row_csv(state['MODEL_OUT_CSV'], ['timestamp', 'p1_id', 'p1_activity', 'p2_id', 'p2_activity'])
                print(f"[TI] Date rollover. New output: {state['MODEL_OUT_CSV']}")

            # 2. Read Data
            dataOk, frameNumber, detObj, timestamp = readAndParseData14xx(DATA, cfg)
            
            # If not recording or no data, just return
            if not gui.is_recording or not dataOk: 
                return

            timestamp_float = timestamp
            pc = np.array(detObj.get('pointCloud', []))
            
            if pc.size == 0:
                gui.update_point_count_quality(0)
                return

            # 3. Log Raw Data (ti.py style - Hardware IDs)
            # Create DF compatible with ti.py format
            df = pd.DataFrame(pc, columns=['x','y','z','doppler','snr','elev','azim','range'])
            df['timestamp'] = timestamp_float
            df['frame'] = frameNumber
            
            # Use hardware target indices if available, else -1
            targetIndices = detObj.get('targetIndices', [])
            if len(targetIndices) == len(pc):
                df['target_id'] = targetIndices
            else:
                df['target_id'] = -1

            # UWB calcs
            df[['dist_uwb', 'azim_uwb', 'elev_uwb']] = df.apply(
                calculate_dist_angle_from_uwb, axis=1, result_type='expand'
            )

            # Reorder columns to match ti.py
            desired_order = [
                'timestamp', 'x', 'y', 'z', 'doppler', 'snr', 'elev', 'azim', 'range', 
                'frame', 'target_id', 'dist_uwb', 'azim_uwb', 'elev_uwb'
            ]
            df_log = df[[c for c in desired_order if c in df.columns]]
            save_df_csv(df_log, state['POINTCLOUD_CSV'])

            # 4. Update Software Tracker (For GUI & Inference)
            # We pass only x,y,z,doppler
            tracked_data = tracker.update(pc[:, :4])

            # 5. Visualization
            gui.update_point_count_quality(len(pc))
            display_points = []
            display_ids = []
            
            for tid, points in tracked_data.items():
                display_points.append(points[:, :3])
                display_ids.extend([tid] * len(points))
            
            if display_points:
                all_disp_pts = np.vstack(display_points)
                gui.update_scatter_with_ids(all_disp_pts, display_ids)
                gui.draw_track_boxes(tracker)
            else:
                gui.update_scatter_with_ids(pc[:,:3], [-1]*len(pc))

            # 6. Inference
            for tid, points in tracked_data.items():
                track = tracker.tracks[tid]
                
                # Sliding window
                frame_features = points.tolist()
                padded_frame = fill_frame(frame_features, FRAME_PAD_TARGET)
                track.buffer.append(padded_frame)
                
                if len(track.buffer) >= WINDOW_SIZE:
                    window_data = track.buffer[-WINDOW_SIZE:]
                    tensor = torch.tensor(window_data, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(tensor)
                        exclude_idx = torch.tensor([7, 8, 9], device=logits.device)
                        masked = logits.clone()
                        masked[:, exclude_idx] = float('-inf')
                        _, pred = torch.max(logits, 1)
                        label = num_to_class(int(pred.item()))
                    
                    track.prediction = label
                    track.buffer = track.buffer[HOP_SIZE:] # Slide
                    
            # 6.5 Log Model Output (Synchronized)
            # Log every HOP_SIZE frames (when inference typically updates)
            if frameNumber % HOP_SIZE == 0:
                 # Use float timestamp with 4 decimal precision like ti.py
                 ts_str = f"{timestamp_float:.4f}"
                 
                 # P1
                 p1_id, p1_act = "NA", "NA"
                 if 1 in tracker.tracks and tracker.tracks[1].missing_frames < MAX_MISSING_FRAMES:
                     p1_id = "1"
                     p1_act = tracker.tracks[1].prediction
                 
                 # P2
                 p2_id, p2_act = "NA", "NA"
                 if 2 in tracker.tracks and tracker.tracks[2].missing_frames < MAX_MISSING_FRAMES:
                     p2_id = "2"
                     p2_act = tracker.tracks[2].prediction
                
                 # Only save if AT LEAST ONE person is detected (non-NA)
                 if p1_id != "NA" or p2_id != "NA":
                     append_row_csv(state['MODEL_OUT_CSV'], [ts_str, p1_id, p1_act, p2_id, p2_act])

            # 7. Update GUI Labels
            if 1 in tracker.tracks and tracker.tracks[1].missing_frames < MAX_MISSING_FRAMES:
                 gui.update_activity(tracker.tracks[1].prediction, is_person1=True)
            else:
                 gui.update_activity("No Subject", is_person1=True)

            if 2 in tracker.tracks and tracker.tracks[2].missing_frames < MAX_MISSING_FRAMES:
                 gui.update_activity(tracker.tracks[2].prediction, is_person1=False)
            else:
                 gui.update_activity("Empty", is_person1=False)

        except Exception as e:
            print(f"Error in loop: {e}")

    timer = QtCore.QTimer()
    timer.timeout.connect(on_tick)
    timer.start(30) # ~33fps
    sys.exit(app.exec_())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TI Radar GUI & Logger")
    parser.add_argument(
        "--shared_data_dir",
        type=str,
        required=True,
        help="The absolute path to the shared_data directory."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=False,
        help="The application version string."
    )
    args = parser.parse_args()

    main(shared_data_dir=args.shared_data_dir)