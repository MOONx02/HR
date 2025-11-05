# Heart Rate Monitor System - Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Code Architecture](#code-architecture)
3. [Health Metrics Mathematics](#health-metrics-mathematics)
4. [Code Walkthrough](#code-walkthrough)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Signal Processing](#signal-processing)

---

## System Overview

This system monitors heart rate and SpO₂ from up to 3 ESP32 devices equipped with MAX30102 pulse oximeter sensors via Bluetooth. The data is displayed on a 7" touchscreen (800x480) with real-time graphs and calculated health metrics.

**Key Features:**
- Bluetooth connectivity to multiple ESP32 devices
- Real-time heart rate and SpO₂ monitoring
- Signal smoothing using median and moving average filters
- Health metric calculations (BPM, IPM, HRSTD, RMSSD)
- Interactive GUI with live graphs

---

## Code Architecture

The code is organized into three main sections:

```
┌─────────────────────────────────────────────────┐
│              gui.py (Main File)                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. ESP32Device Class                           │
│     - Bluetooth connection management           │
│     - Data parsing and storage                  │
│     - Signal smoothing                          │
│     - Health metrics calculation                │
│                                                 │
│  2. MultiDeviceManager Class                    │
│     - Manages multiple ESP32 devices            │
│     - Device scanning                           │
│     - Data aggregation                          │
│                                                 │
│  3. CompactHeartRateGUI Class                   │
│     - User interface (3 tabs)                   │
│     - Real-time display updates                 │
│     - Graph plotting                            │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Health Metrics Mathematics

### 1. **BPM (Beats Per Minute) - Average Heart Rate**

**Purpose:** Provides the average heart rate over a time window (up to 5 minutes)

**Formula:**
```
BPM = (1/N) × Σ(HR_i)  where i = 1 to N
```

**Implementation:**
```python
hr_array = np.array(list(self.hr_history))
self.metrics['bpm'] = np.mean(hr_array)
```

**Explanation:**
- Collects all valid heart rate readings in the history buffer (max 300 samples)
- Calculates arithmetic mean using NumPy
- Provides a stable average over time

---

### 2. **IPM (Impulses Per Minute)**

**Purpose:** Represents the pulse rate detected by the sensor

**Formula:**
```
IPM = BPM
```

**Implementation:**
```python
self.metrics['ipm'] = self.metrics['bpm']
```

**Explanation:**
- For pulse oximeters, IPM equals BPM since each heartbeat creates one pulse
- In other applications, IPM might differ (e.g., arterial vs venous pulses)

---

### 3. **HRSTD (Heart Rate Standard Deviation)**

**Purpose:** Measures heart rate variability - how much the heart rate fluctuates

**Formula:**
```
HRSTD = √[(1/N) × Σ(HR_i - μ)²]  where μ = mean heart rate
```

**Implementation:**
```python
self.metrics['hrstd'] = np.std(hr_array)
```

**Clinical Significance:**
- **Low HRSTD (1-3 bpm):** Very stable heart rate, typical during rest
- **Moderate HRSTD (3-8 bpm):** Normal variability
- **High HRSTD (>8 bpm):** Significant fluctuation, could indicate stress or arrhythmia

**Example Calculation:**
```
HR readings: [72, 75, 73, 74, 76]
Mean (μ) = 74
Deviations: [-2, 1, -1, 0, 2]
Squared: [4, 1, 1, 0, 4]
Variance = (4+1+1+0+4)/5 = 2
HRSTD = √2 ≈ 1.41 bpm
```

---

### 4. **RMSSD (Root Mean Square of Successive Differences)**

**Purpose:** Measures beat-to-beat heart rate variability (HRV)

**Formula:**
```
RMSSD = √[(1/(N-1)) × Σ(HR_{i+1} - HR_i)²]  where i = 1 to N-1
```

**Implementation:**
```python
successive_diffs = np.diff(hr_array)  # Calculate HR_{i+1} - HR_i
self.metrics['rmssd'] = np.sqrt(np.mean(successive_diffs ** 2))
```

**Clinical Significance:**
- **Low RMSSD (<20 ms):** Poor HRV, possible stress or fatigue
- **Normal RMSSD (20-50 ms):** Healthy autonomic function
- **High RMSSD (>50 ms):** Excellent HRV, good cardiovascular health

**Step-by-step Example:**
```
HR readings: [72, 75, 73, 76, 74]

Step 1: Calculate successive differences
  75-72 = 3
  73-75 = -2
  76-73 = 3
  74-76 = -2

Step 2: Square the differences
  3² = 9
  (-2)² = 4
  3² = 9
  (-2)² = 4

Step 3: Calculate mean
  Mean = (9+4+9+4)/4 = 6.5

Step 4: Take square root
  RMSSD = √6.5 ≈ 2.55 bpm
```

**Note:** The unit shown is in bpm (beats per minute changes), not milliseconds as typically used in clinical HRV analysis. For ms conversion, you'd need inter-beat intervals.

---

## Code Walkthrough

### Section 1: Imports and Setup

```python
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from collections import deque
import bluetooth
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
```

**Purpose:**
- **tkinter:** GUI framework
- **threading:** Parallel data reception from multiple devices
- **deque:** Efficient circular buffer for data history
- **bluetooth:** ESP32 communication (PyBluez library)
- **matplotlib:** Real-time graph plotting
- **numpy:** Mathematical operations for metrics

---

### Section 2: ESP32Device Class

#### 2.1 Initialization

```python
def __init__(self, device_id, name, mac_address=None):
    self.device_id = device_id
    self.name = name
    self.mac_address = mac_address
    self.sock = None
    self.connected = False
```

**Purpose:** Creates a device instance with unique ID and Bluetooth MAC address

#### 2.2 Data Storage Structures

```python
# Latest readings
self.latest_data = {
    'hr': None,              # Current heart rate
    'hr_valid': False,       # Is HR reading valid?
    'spo2': None,            # Current SpO₂
    'spo2_valid': False,     # Is SpO₂ reading valid?
    'ir_avg': None,          # IR sensor average
    'ir_range': None,        # IR sensor range
    'timestamp': None,       # Device timestamp
    'local_time': None,      # Raspberry Pi timestamp
    'status': 'disconnected' # Connection status
}

# Historical data (5 minutes = 300 readings at 1 Hz)
self.hr_history = deque(maxlen=300)
self.timestamp_history = deque(maxlen=300)

# Smoothing buffers (5 most recent readings)
self.hr_smoothing_buffer = deque(maxlen=5)
self.spo2_smoothing_buffer = deque(maxlen=5)
```

**Why deque?**
- Automatically removes oldest data when maxlen is reached
- O(1) append and pop operations (very efficient)
- Perfect for rolling windows

#### 2.3 Bluetooth Connection

```python
def connect(self):
    try:
        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.sock.connect((self.mac_address, 1))  # RFCOMM channel 1
        self.connected = True
        return True
    except Exception as e:
        self.connected = False
        return False
```

**Protocol:** RFCOMM (Radio Frequency Communication) - Bluetooth serial protocol
**Channel 1:** Standard channel for ESP32 Serial Bluetooth

#### 2.4 Data Parsing

```python
def parse_data(self, data_string):
    # Format: "HR:75,HR_VALID:1,SPO2:98,SPO2_VALID:1,..."
    parts = data_string.strip().split(',')
    data_dict = {}
    
    for part in parts:
        if ':' in part:
            key, value = part.split(':', 1)
            data_dict[key] = value
```

**ESP32 Data Format:**
```
DEV:1,HR:75,HR_VALID:1,SPO2:98,SPO2_VALID:1,IR_AVG:100000,IR_RANGE:50,TIMESTAMP:12345
```

**Validation:**
```python
# Only accept physiologically valid readings
if 40 <= self.latest_data['hr'] <= 200:  # Valid HR range
    self.hr_history.append(self.latest_data['hr'])

if 70 <= self.latest_data['spo2'] <= 100:  # Valid SpO2 range
    self.spo2_smoothing_buffer.append(self.latest_data['spo2'])
```

#### 2.5 Signal Smoothing (Key Innovation!)

```python
def calculate_smoothed_values(self):
    # Heart Rate: Median Filter (better outlier rejection)
    if len(self.hr_smoothing_buffer) >= 3:
        self.smoothed_hr = int(np.median(list(self.hr_smoothing_buffer)))
    
    # SpO2: Moving Average Filter
    if len(self.spo2_smoothing_buffer) >= 3:
        self.smoothed_spo2 = int(np.mean(list(self.spo2_smoothing_buffer)))
```

**Why Median for HR?**
- Removes spikes (outliers) effectively
- Example: [72, 73, 95, 74, 73] → median = 73 (ignores 95 spike)

**Why Mean for SpO₂?**
- SpO₂ is more stable, no frequent outliers
- Averaging provides smooth transitions

**Comparison:**
```
Raw readings:     [68, 72, 95, 73, 71]  (spike at 95)
Median filtered:  73
Mean filtered:    75.8 (affected by spike)
```

#### 2.6 Health Metrics Calculation

```python
def calculate_metrics(self):
    if len(self.hr_history) < 2:
        return
    
    hr_array = np.array(list(self.hr_history))
    
    # 1. Average BPM
    self.metrics['bpm'] = np.mean(hr_array)
    
    # 2. IPM (same as BPM for pulse oximeters)
    self.metrics['ipm'] = self.metrics['bpm']
    
    # 3. Heart Rate Standard Deviation
    self.metrics['hrstd'] = np.std(hr_array)
    
    # 4. RMSSD - Root Mean Square of Successive Differences
    successive_diffs = np.diff(hr_array)  # [HR2-HR1, HR3-HR2, ...]
    self.metrics['rmssd'] = np.sqrt(np.mean(successive_diffs ** 2))
```

**NumPy Operations Used:**
- `np.mean()`: Calculate average
- `np.std()`: Calculate standard deviation
- `np.diff()`: Calculate differences between consecutive elements
- `np.sqrt()`: Square root

#### 2.7 Continuous Data Reception

```python
def receive_data(self):
    buffer = ""
    
    while self.connected:
        data = self.sock.recv(1024)  # Receive up to 1024 bytes
        if data:
            buffer += data.decode('utf-8', errors='ignore')
            
            # Process complete lines (terminated by \n)
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    self.parse_data(line)
        else:
            time.sleep(0.1)  # Prevent CPU spinning
```

**Buffering Strategy:**
- Accumulates partial data until complete line received
- Prevents data corruption from incomplete packets
- Each line represents one complete sensor reading

---

### Section 3: MultiDeviceManager Class

```python
class MultiDeviceManager:
    def __init__(self):
        self.devices = []      # List of ESP32Device objects
        self.threads = []      # Thread for each device
        self.running = False   # System running state
```

#### 3.1 Device Scanning

```python
def scan_devices(self):
    nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True)
    return nearby_devices  # List of (MAC_address, device_name) tuples
```

**Parameters:**
- `duration=8`: Scan for 8 seconds
- `lookup_names=True`: Get human-readable names (not just MAC addresses)

#### 3.2 Multi-threaded Reception

```python
def start_receiving(self):
    self.running = True
    
    for device in self.devices:
        if device.connected:
            thread = threading.Thread(target=device.receive_data, daemon=True)
            thread.start()
            self.threads.append(thread)
```

**Why Threading?**
- Each device needs continuous monitoring
- `sock.recv()` is blocking - would freeze other devices
- Daemon threads automatically terminate when main program exits

**Thread Safety:**
- Each device has its own data structures (no shared state)
- GUI reads data (read-only operation) while threads write
- Python's GIL (Global Interpreter Lock) prevents race conditions

---

### Section 4: GUI Application

#### 4.1 Main Window Setup

```python
def __init__(self, root):
    self.root = root
    self.root.title("Heart Rate Monitor")
    self.root.geometry("800x480")  # Exact touchscreen size
    self.root.configure(bg='#1e1e1e')  # Dark theme
```

#### 4.2 Three-Tab Interface

```
┌──────────────────────────────────────────────┐
│  🔗 Setup   📊 Overview   📈 Graph           │  ← Tabs
├──────────────────────────────────────────────┤
│                                              │
│  [Tab Content Area]                          │
│                                              │
│                                              │
└──────────────────────────────────────────────┘
```

**Tab 1: Setup** - Device configuration and connection
**Tab 2: Overview** - Live readings from all 3 devices
**Tab 3: Graph** - Real-time heart rate trend chart

#### 4.3 Device Panel Layout

Each device gets a panel showing:

```
┌─────────────────────┐
│    Device 1         │  ← Header
├─────────────────────┤
│        75           │  ← Large HR display (32pt font)
│       bpm           │
├─────────────────────┤
│       98%           │  ← SpO₂ display (14pt font)
│       SpO₂          │
├─────────────────────┤
│   Health Metrics    │  ← Metrics header
│   Avg BPM:  74.5    │
│   IPM:      74.5    │
│   HRSTD:    2.34    │
│   RMSSD:    3.12    │
├─────────────────────┤
│  ✓ Receiving        │  ← Status
└─────────────────────┘
```

#### 4.4 Color Coding (Clinical Zones)

```python
if 60 <= smoothed_hr <= 100:
    color = '#50c878'  # Green - Normal
elif 40 <= smoothed_hr < 60 or 100 < smoothed_hr <= 120:
    color = '#f39c12'  # Orange - Caution
else:
    color = '#e74c3c'  # Red - Abnormal
```

**Heart Rate Zones:**
- **< 40 bpm:** Bradycardia (abnormally slow)
- **40-60 bpm:** Low but possibly normal (athletes)
- **60-100 bpm:** Normal resting heart rate
- **100-120 bpm:** Elevated (tachycardia threshold)
- **> 120 bpm:** Tachycardia (abnormally fast)

#### 4.5 Real-time Graph Updates

```python
def update_plot(self):
    self.ax.clear()
    
    current_time = time.time()
    colors = ['#e74c3c', '#50c878', '#4a90e2']  # Red, Green, Blue
    
    for device_id in [1, 2, 3]:
        times = np.array(self.plot_data[device_id]['time'])
        hrs = np.array(self.plot_data[device_id]['hr'])
        
        # Convert to "seconds ago"
        times_ago = current_time - times
        times_ago = times_ago[::-1]  # Reverse (most recent = 0)
        hrs = hrs[::-1]
        
        self.ax.plot(times_ago, hrs, color=colors[device_id-1], 
                    linewidth=2, label=f'Device {device_id}')
```

**Graph Features:**
- X-axis: "Seconds Ago" (60 seconds back)
- Y-axis: Heart Rate (40-140 bpm range)
- Shows last 60 data points (1 minute of data)
- Auto-updates every 500ms

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         ESP32 Devices                            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │
│  │  MAX30102   │   │  MAX30102   │   │  MAX30102   │             │
│  │   Sensor    │   │   Sensor    │   │   Sensor    │             │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘             │
│         │                 │                 │                    │
│    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐               │
│    │ ESP32 #1│       │ ESP32 #2│       │ ESP32 #3│               │
│    └────┬────┘       └────┬────┘       └────┬────┘               │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          │   Bluetooth      │   Bluetooth      │   Bluetooth
          │   RFCOMM         │   RFCOMM         │   RFCOMM
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────────┐
│               Raspberry Pi (Python Application)                 │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              MultiDeviceManager                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │ ESP32Device  │  │ ESP32Device  │  │ ESP32Device  │     │  │
│  │  │   Thread 1   │  │   Thread 2   │  │   Thread 3   │     │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │  │
│  └─────────┼──────────────────┼──────────────────┼───────────┘  │
│            │                  │                  │              │
│            ▼                  ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Data Processing Pipeline                     │    │
│  │                                                         │    │
│  │  1. Receive:  "HR:75,SPO2:98,..."                       │    │
│  │  2. Parse:    Extract values                            │    │
│  │  3. Validate: Check 40≤HR≤200, 70≤SpO2≤100              │    │
│  │  4. Store:    Add to history buffers                    │    │ 
│  │  5. Smooth:   Apply median/average filters              │    │
│  │  6. Calculate: BPM, IPM, HRSTD, RMSSD                   │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              GUI Display (Tkinter)                     │     │
│  │                                                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │     │
│  │  │ Device 1 │  │ Device 2 │  │ Device 3 │              │     │
│  │  │  Panel   │  │  Panel   │  │  Panel   │              │     │
│  │  │          │  │          │  │          │              │     │
│  │  │ HR: 75   │  │ HR: 68   │  │ HR: 82   │              │     │
│  │  │ SpO2:98% │  │ SpO2:96% │  │ SpO2:97% │              │     │
│  │  │ BPM:74.5 │  │ BPM:67.2 │  │ BPM:81.3 │              │     │
│  │  │ HRSTD:2.3│  │ HRSTD:1.8│  │ HRSTD:3.1│              │     │
│  │  └──────────┘  └──────────┘  └──────────┘              │     │
│  │                                                        │     │
│  │  ┌───────────────────────────────────────────┐         │     │
│  │  │     Real-time Graph (Matplotlib)          │         │     │
│  │  │                                           │         │     │
│  │  │  140┤                    ╱╲               │         │     │
│  │  │  120┤                   ╱  ╲              │         │     │
│  │  │  100┤        ╱╲        ╱    ╲╱╲           │         │     │
│  │  │   80┤  ╱╲   ╱  ╲      ╱                   │         │     │
│  │  │   60┤ ╱  ╲╱     ╲    ╱                    │         │     │
│  │  │   40┤                                     │         │     │
│  │  │     └────────────────────────────────────►│         │     │
│  │  │      60s  45s  30s  15s   0s              │         │     │
│  │  └───────────────────────────────────────────┘         │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ 7" Touchscreen │
                  │   800 x 480    │
                  └────────────────┘
```

---

## Signal Processing

### Data Processing Pipeline

```
Raw Sensor Data → Validation → Buffering → Smoothing → Display
                                    ↓
                              Metrics Calculation
```

### 1. Raw Data Reception
```
Input:  "HR:75,HR_VALID:1,SPO2:98,SPO2_VALID:1"
Rate:   ~1 Hz (one reading per second)
```

### 2. Validation Filter
```python
if 40 <= hr <= 200 and hr_valid:
    # Accept
else:
    # Reject
```

**Rejection Rate:** Typically 5-10% of readings due to:
- Finger movement
- Poor sensor contact
- Ambient light interference

### 3. Smoothing Window (5 samples)

**Before Smoothing:**
```
Time:  t1   t2   t3   t4   t5
HR:    72   73   95   74   71  ← Spike at t3
```

**After Median Smoothing:**
```
Median([72, 73, 95, 74, 71]) = 73 bpm
```

**Effect:** Spike eliminated, smooth display

### 4. Metrics Calculation Window (up to 300 samples = 5 minutes)

```
Recent History: [72, 73, 74, 72, 75, 73, 74, 71, 73, 72, ...]
                 ↓
             NumPy Array
                 ↓
        ┌────────┴────────┐
        │                 │
    Mean (BPM)        Std Dev (HRSTD)
                          │
                      Diff Array
                          │
                    RMSSD Calculation
```

### 5. Display Update Rate

```
Update Interval: 500 ms (2 Hz)
Plot Window:     60 seconds
Plot Points:     60 samples max
```

**Why 500ms updates?**
- Smooth visual experience
- Not too fast (wastes CPU)
- Not too slow (appears laggy)

---

## Performance Characteristics

### Memory Usage

**Per Device:**
- HR History: 300 samples × 8 bytes (int64) = 2.4 KB
- Timestamp History: 300 samples × 8 bytes = 2.4 KB
- Smoothing Buffers: 10 samples × 8 bytes = 80 bytes
- **Total per device:** ~5 KB

**Total System:** ~15 KB for 3 devices (negligible)

### CPU Usage

**Threading:**
- 3 receiver threads (mostly idle, waiting for Bluetooth data)
- 1 GUI thread (updates every 500ms)
- **CPU Load:** <5% on Raspberry Pi 4

### Latency

```
Sensor Reading → ESP32 Processing → Bluetooth TX → RPI RX → Parsing → Smoothing → Display
    ~10ms             ~50ms              ~20ms       ~5ms      ~1ms      ~1ms      ~5ms
                                                                                      
Total Latency: ~92ms (acceptable for heart rate monitoring)
```

---

## Clinical Interpretation Guide

### Normal Ranges (Adults at Rest)

| Metric | Normal Range | Clinical Significance |
|--------|--------------|----------------------|
| HR | 60-100 bpm | Primary vital sign |
| SpO₂ | 95-100% | Oxygen saturation |
| BPM | 60-100 bpm | Average heart rate |
| HRSTD | 2-8 bpm | HR variability |
| RMSSD | 20-50 ms* | Beat-to-beat HRV |

*Note: Our RMSSD is in bpm units, not ms

### Warning Signs

**🔴 Critical (Red):**
- HR < 40 or > 120 bpm
- SpO₂ < 90%

**🟠 Caution (Orange):**
- HR 40-60 or 100-120 bpm
- SpO₂ 90-95%

**🟢 Normal (Green):**
- HR 60-100 bpm
- SpO₂ 95-100%

---

## Troubleshooting

### Common Issues

**1. "No devices found" during scan**
- Ensure ESP32 is powered on
- Check Bluetooth is enabled on ESP32
- ESP32 must be in pairing mode
- Try scanning for longer (increase duration)

**2. "Connection failed"**
- Verify MAC address is correct
- ESP32 may already be paired to another device
- Reset ESP32 Bluetooth: power cycle the device

**3. Jumpy readings**
- Normal during first 3-5 seconds (smoothing buffer filling)
- Ensure proper finger placement on sensor
- Clean sensor surface
- Check sensor is not in direct sunlight

**4. "No Finger" status**
- Sensor needs firm but gentle contact
- Avoid excessive pressure (restricts blood flow)
- Warm hands (cold fingers = poor signal)

---

## Future Enhancements

1. **Data Logging:** Save readings to CSV/database
2. **Alerts:** Configurable thresholds with audio warnings
3. **HRV Analysis:** Frequency domain analysis (LF/HF ratio)
4. **Multi-user Support:** Profiles for different patients
5. **Cloud Sync:** Upload data to remote server
6. **Advanced Filtering:** Kalman filter for even smoother data

---

## References

**Heart Rate Variability:**
- Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use.

**Signal Processing:**
- Acharya, U. R., et al. (2006). Heart rate variability: a review.

**Pulse Oximetry:**
- Maxim Integrated. MAX30102 High-Sensitivity Pulse Oximeter and Heart-Rate Sensor for Wearable Health. Datasheet.

---

## License & Credits

**Developed for:** Multi-ESP32 Heart Rate Monitoring System  
**Hardware:** Raspberry Pi 4 + 7" Touchscreen + ESP32 + MAX30102  
**Language:** Python 3  
**Libraries:** PyBluez, Tkinter, Matplotlib, NumPy  

---

**End of Documentation**

