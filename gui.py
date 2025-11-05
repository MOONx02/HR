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


# ============================================================================
# ESP32 Device Management Classes
# ============================================================================

class ESP32Device:
    """Represents a single ESP32 heart rate monitor device"""
    
    def __init__(self, device_id, name, mac_address=None):
        self.device_id = device_id
        self.name = name
        self.mac_address = mac_address
        self.sock = None
        self.connected = False
        
        # Data storage
        self.latest_data = {
            'hr': None,
            'hr_valid': False,
            'spo2': None,
            'spo2_valid': False,
            'ir_avg': None,
            'ir_range': None,
            'timestamp': None,
            'local_time': None,
            'status': 'disconnected'
        }
        
        # Historical data for metrics calculation (store last 5 minutes = 300 readings)
        self.hr_history = deque(maxlen=300)
        self.timestamp_history = deque(maxlen=300)
        
        # Smoothing buffer for display (last 5 readings for moving average)
        self.hr_smoothing_buffer = deque(maxlen=5)
        self.spo2_smoothing_buffer = deque(maxlen=5)
        self.smoothed_hr = None
        self.smoothed_spo2 = None
        
        # Calculated metrics
        self.metrics = {
            'bpm': None,
            'ipm': None,
            'hrstd': None,
            'rmssd': None,
            'avg_spo2': None
        }
    
    def connect(self):
        """Connect to the ESP32 device via Bluetooth"""
        if self.mac_address is None:
            print(f"[{self.name}] MAC address not set.")
            return False
        
        try:
            print(f"[{self.name}] Connecting to {self.mac_address}...")
            self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.sock.connect((self.mac_address, 1))
            self.connected = True
            self.latest_data['status'] = 'connected'
            print(f"[{self.name}] ✅ Connected successfully!")
            return True
        except Exception as e:
            print(f"[{self.name}] ❌ Connection failed: {e}")
            self.connected = False
            self.latest_data['status'] = 'error'
            return False
    
    def disconnect(self):
        """Disconnect from the ESP32 device"""
        if self.sock:
            try:
                self.sock.close()
                print(f"[{self.name}] Disconnected")
            except:
                pass
        self.connected = False
        self.latest_data['status'] = 'disconnected'
    
    def parse_data(self, data_string):
        """Parse incoming data from ESP32"""
        try:
            parts = data_string.strip().split(',')
            data_dict = {}
            
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    data_dict[key] = value
            
            # Update latest data
            if 'HR' in data_dict:
                self.latest_data['hr'] = int(data_dict['HR'])
            if 'HR_VALID' in data_dict:
                self.latest_data['hr_valid'] = bool(int(data_dict['HR_VALID']))
            if 'SPO2' in data_dict:
                self.latest_data['spo2'] = int(data_dict['SPO2'])
            if 'SPO2_VALID' in data_dict:
                self.latest_data['spo2_valid'] = bool(int(data_dict['SPO2_VALID']))
            if 'IR_AVG' in data_dict:
                self.latest_data['ir_avg'] = int(data_dict['IR_AVG'])
            if 'IR_RANGE' in data_dict:
                self.latest_data['ir_range'] = int(data_dict['IR_RANGE'])
            if 'TIMESTAMP' in data_dict:
                self.latest_data['timestamp'] = int(data_dict['TIMESTAMP'])
            
            self.latest_data['local_time'] = datetime.now()
            
            # Check for status messages
            if 'STATUS' in data_dict:
                self.latest_data['status'] = data_dict['STATUS']
            else:
                self.latest_data['status'] = 'receiving'
            
            # Add to history if heart rate is valid
            if self.latest_data['hr_valid'] and self.latest_data['hr'] is not None:
                if 40 <= self.latest_data['hr'] <= 200:
                    self.hr_history.append(self.latest_data['hr'])
                    self.timestamp_history.append(time.time())
                    self.hr_smoothing_buffer.append(self.latest_data['hr'])
            
            # Add SpO2 to smoothing buffer if valid
            if self.latest_data['spo2_valid'] and self.latest_data['spo2'] is not None:
                if 70 <= self.latest_data['spo2'] <= 100:
                    self.spo2_smoothing_buffer.append(self.latest_data['spo2'])
            
            # Calculate smoothed values
            self.calculate_smoothed_values()
            
            # Calculate metrics
            self.calculate_metrics()
            
            return True
        except Exception as e:
            print(f"[{self.name}] Parse error: {e}")
            return False
    
    def calculate_smoothed_values(self):
        """Calculate smoothed values using moving average filter"""
        # Smooth heart rate - use median of last 5 readings (better outlier rejection)
        if len(self.hr_smoothing_buffer) >= 3:
            self.smoothed_hr = int(np.median(list(self.hr_smoothing_buffer)))
        elif len(self.hr_smoothing_buffer) > 0:
            self.smoothed_hr = int(np.mean(list(self.hr_smoothing_buffer)))
        
        # Smooth SpO2 - use average of last 5 readings
        if len(self.spo2_smoothing_buffer) >= 3:
            self.smoothed_spo2 = int(np.mean(list(self.spo2_smoothing_buffer)))
        elif len(self.spo2_smoothing_buffer) > 0:
            self.smoothed_spo2 = int(list(self.spo2_smoothing_buffer)[-1])
    
    def calculate_metrics(self):
        """Calculate health metrics from historical data"""
        if len(self.hr_history) < 2:
            return
        
        hr_array = np.array(list(self.hr_history))
        
        # BPM - Average heart rate
        self.metrics['bpm'] = np.mean(hr_array)
        
        # IPM - Impulses Per Minute
        self.metrics['ipm'] = self.metrics['bpm']
        
        # HRSTD - Heart Rate Standard Deviation
        self.metrics['hrstd'] = np.std(hr_array)
        
        # RMSSD - Root Mean Square of Successive Differences
        if len(hr_array) >= 2:
            successive_diffs = np.diff(hr_array)
            self.metrics['rmssd'] = np.sqrt(np.mean(successive_diffs ** 2))
        
        # Average SpO2 if valid
        if self.latest_data['spo2_valid'] and self.latest_data['spo2'] is not None:
            self.metrics['avg_spo2'] = self.latest_data['spo2']
    
    def receive_data(self):
        """Continuously receive data from the device"""
        buffer = ""
        
        while self.connected:
            try:
                data = self.sock.recv(1024)
                if data:
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            self.parse_data(line)
                else:
                    time.sleep(0.1)
            except bluetooth.BluetoothError as e:
                print(f"[{self.name}] Bluetooth error: {e}")
                self.connected = False
                self.latest_data['status'] = 'error'
                break
            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                time.sleep(1)


class MultiDeviceManager:
    """Manages multiple ESP32 devices"""
    
    def __init__(self):
        self.devices = []
        self.threads = []
        self.running = False
    
    def add_device(self, device_id, name, mac_address=None):
        """Add a device to manage"""
        device = ESP32Device(device_id, name, mac_address)
        self.devices.append(device)
        return device
    
    def scan_devices(self):
        """Scan for nearby Bluetooth devices"""
        print("\n🔍 Scanning for Bluetooth devices...")
        
        try:
            nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True)
            print(f"Found {len(nearby_devices)} devices")
            return nearby_devices
        except Exception as e:
            print(f"❌ Scan failed: {e}")
            return []
    
    def connect_all(self):
        """Connect to all devices"""
        for device in self.devices:
            if device.mac_address:
                device.connect()
                time.sleep(1)
    
    def start_receiving(self):
        """Start receiving data from all connected devices"""
        self.running = True
        
        for device in self.devices:
            if device.connected:
                thread = threading.Thread(target=device.receive_data, daemon=True)
                thread.start()
                self.threads.append(thread)
    
    def stop(self):
        """Stop all devices and threads"""
        self.running = False
        for device in self.devices:
            device.disconnect()
    
    def get_all_data(self):
        """Get data from all devices"""
        return {device.device_id: {
            'name': device.name,
            'data': device.latest_data,
            'metrics': device.metrics,
            'connected': device.connected,
            'smoothed_hr': device.smoothed_hr,
            'smoothed_spo2': device.smoothed_spo2
        } for device in self.devices}


# ============================================================================
# GUI Application
# ============================================================================

class CompactHeartRateGUI:
    """Compact GUI Application for 800x480 Display"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Rate Monitor")
        self.root.geometry("800x480")
        self.root.configure(bg='#1e1e1e')
        
        # Manager for ESP32 devices
        self.manager = None
        self.running = False
        
        # Store scanned devices
        self.scanned_devices = []
        
        # Data for plotting
        self.plot_data = {
            1: {'time': [], 'hr': []},
            2: {'time': [], 'hr': []},
            3: {'time': [], 'hr': []}
        }
        self.max_plot_points = 60  # Show last 60 seconds
        
        # Setup UI
        self.setup_ui()
        
        # Update timer
        self.update_interval = 500  # ms
        self.update_display()
    
    def setup_ui(self):
        """Setup the compact user interface"""
        
        # Title bar - very compact
        title_frame = tk.Frame(self.root, bg='#2d2d2d', height=35)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="❤️ Wireless Pulse Monitor",
            font=('Arial', 12, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=10)
        
        # Status indicator
        self.status_label = tk.Label(
            title_frame,
            text="⚪ Not Connected",
            font=('Arial', 8),
            bg='#2d2d2d',
            fg='#888888'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Create tab control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Style the notebook for compact display
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#1e1e1e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#2d2d2d', foreground='white', 
                       padding=[15, 5], font=('Arial', 9, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#4a90e2')],
                 foreground=[('selected', 'white')])
        
        # Create tabs
        self.setup_connection_tab()
        self.setup_overview_tab()
        self.setup_graph_tab()
    
    def setup_connection_tab(self):
        """Setup the connection configuration tab"""
        conn_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(conn_tab, text='🔗 Setup')
        
        # Center container
        center_frame = tk.Frame(conn_tab, bg='#1e1e1e')
        center_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        # Instructions
        tk.Label(
            center_frame,
            text="Device Configuration",
            font=('Arial', 12, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        ).pack(pady=2)
        
        # Configuration panel
        config_panel = tk.Frame(center_frame, bg='#2d2d2d', relief=tk.RAISED, borderwidth=2)
        config_panel.pack(pady=2, padx=20)
        
        # MAC entries - vertical layout for easier touch input
        self.mac_entries = {}
        for i in range(1, 4):
            entry_frame = tk.Frame(config_panel, bg='#2d2d2d')
            entry_frame.pack(pady=3, padx=10, fill=tk.X)
            
            tk.Label(
                entry_frame,
                text=f"ESP32 Device {i}:",
                font=('Arial', 10),
                bg='#2d2d2d',
                fg='#ffffff',
                width=15,
                anchor=tk.W
            ).pack(side=tk.LEFT, padx=5)
            
            entry = tk.Entry(entry_frame, font=('Arial', 10), width=25,
                           bg='#3d3d3d', fg='white', insertbackground='white')
            entry.pack(side=tk.LEFT, padx=5)
            self.mac_entries[i] = entry
        
        # Scan button
        self.scan_btn = tk.Button(
            config_panel,
            text="🔍 Scan for Devices",
            command=self.scan_devices,
            font=('Arial', 10, 'bold'),
            bg='#4a90e2',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.scan_btn.pack(pady=3)
        
        # Scan results display with scrollbar and fixed height
        results_container = tk.Frame(config_panel, bg='#2d2d2d')
        results_container.pack(pady=3, padx=10, fill=tk.X)
        
        # Create canvas for scrolling
        results_canvas = tk.Canvas(results_container, bg='#2d2d2d', height=80, highlightthickness=0)
        results_scrollbar = tk.Scrollbar(results_container, orient="vertical", command=results_canvas.yview)
        self.scan_results_frame = tk.Frame(results_canvas, bg='#2d2d2d')
        
        self.scan_results_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        
        results_canvas.create_window((0, 0), window=self.scan_results_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons - always visible at bottom
        btn_frame = tk.Frame(config_panel, bg='#2d2d2d')
        btn_frame.pack(pady=3)
        
        self.connect_btn = tk.Button(
            btn_frame,
            text="🔗 Connect",
            command=self.connect_devices,
            font=('Arial', 10, 'bold'),
            bg='#50c878',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = tk.Button(
            btn_frame,
            text="⏹ Disconnect",
            command=self.disconnect_devices,
            font=('Arial', 10, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=15,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2,
            state=tk.DISABLED
        )
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
    
    def setup_overview_tab(self):
        """Setup the overview tab showing all devices"""
        overview_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(overview_tab, text='📊 Overview')
        
        # Main data display - 3 columns for 3 devices
        data_container = tk.Frame(overview_tab, bg='#1e1e1e')
        data_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create 3 device panels side by side
        self.device_frames = {}
        for i in range(1, 4):
            frame = self.create_compact_device_panel(data_container, i)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            self.device_frames[i] = frame
    
    def setup_graph_tab(self):
        """Setup the graph tab showing heart rate trends"""
        graph_tab = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(graph_tab, text='📈 Graph')
        
        # Graph frame
        graph_frame = tk.Frame(graph_tab, bg='#2d2d2d', relief=tk.RAISED, borderwidth=1)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(
            graph_frame,
            text="Real-Time Heart Rate Trends",
            font=('Arial', 12, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        ).pack(pady=5)
        
        # Create matplotlib figure - larger for dedicated graph tab
        self.fig = Figure(figsize=(7.5, 3.5), dpi=100, facecolor='#2d2d2d')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('Seconds Ago', color='white', fontsize=10)
        self.ax.set_ylabel('Heart Rate (bpm)', color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_compact_device_panel(self, parent, device_id):
        """Create a very compact panel for one device"""
        
        frame = tk.Frame(parent, bg='#2d2d2d', relief=tk.RAISED, borderwidth=1)
        
        # Header
        tk.Label(
            frame,
            text=f"Device {device_id}",
            font=('Arial', 9, 'bold'),
            bg='#2d2d2d',
            fg='#4a90e2'
        ).pack(pady=2)
        
        # Heart Rate - BIG
        hr_label = tk.Label(
            frame,
            text="--",
            font=('Arial', 32, 'bold'),
            bg='#2d2d2d',
            fg='#50c878'
        )
        hr_label.pack(pady=5)
        
        tk.Label(
            frame,
            text="bpm",
            font=('Arial', 8),
            bg='#2d2d2d',
            fg='#888888'
        ).pack()
        
        # SpO2
        spo2_label = tk.Label(
            frame,
            text="-- %",
            font=('Arial', 14, 'bold'),
            bg='#2d2d2d',
            fg='#4a90e2'
        )
        spo2_label.pack(pady=5)
        
        tk.Label(
            frame,
            text="SpO₂",
            font=('Arial', 7),
            bg='#2d2d2d',
            fg='#888888'
        ).pack()
        
        # Separator
        tk.Frame(frame, bg='#555555', height=1).pack(fill=tk.X, padx=5, pady=5)
        
        # Metrics section
        tk.Label(
            frame,
            text="Health Metrics",
            font=('Arial', 10, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        ).pack()
        
        metrics_container = tk.Frame(frame, bg='#2d2d2d')
        metrics_container.pack(fill=tk.X, padx=5, pady=2)
        
        metrics_labels = {}
        
        for metric_name, metric_label in [
            ('bpm', 'Avg BPM'),
            ('ipm', 'IPM'),
            ('hrstd', 'HRSTD'),
            ('rmssd', 'RMSSD')
        ]:
            metric_frame = tk.Frame(metrics_container, bg='#2d2d2d')
            metric_frame.pack(fill=tk.X)
            
            tk.Label(
                metric_frame,
                text=f"{metric_label}:",
                font=('Arial', 9),
                bg='#2d2d2d',
                fg='#aaaaaa',
                anchor=tk.W
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            label = tk.Label(
                metric_frame,
                text="--",
                font=('Arial', 9, 'bold'),
                bg='#2d2d2d',
                fg='#ffffff',
                anchor=tk.E
            )
            label.pack(side=tk.RIGHT)
            metrics_labels[metric_name] = label
        
        # Status
        status_label = tk.Label(
            frame,
            text="Not Connected",
            font=('Arial', 7),
            bg='#2d2d2d',
            fg='#e74c3c'
        )
        status_label.pack(pady=2)
        
        # Store references
        frame.hr_label = hr_label
        frame.spo2_label = spo2_label
        frame.status_label = status_label
        frame.metrics_labels = metrics_labels
        
        return frame
    
    def scan_devices(self):
        """Scan for Bluetooth devices"""
        self.status_label.config(text="🔍 Scanning...", fg='#f39c12')
        self.scan_btn.config(state=tk.DISABLED, text="Scanning...")
        
        # Clear previous scan results
        for widget in self.scan_results_frame.winfo_children():
            widget.destroy()
        
        def scan_thread():
            try:
                temp_manager = MultiDeviceManager()
                devices = temp_manager.scan_devices()
                
                if devices:
                    # Store scanned devices
                    self.scanned_devices = devices
                    
                    # Display results in scan results frame
                    self.root.after(0, lambda: self.display_scan_results(devices))
                    
                    self.status_label.config(text=f"✓ Found {len(devices)} device(s)", fg='#50c878')
                else:
                    self.scanned_devices = []
                    self.status_label.config(text="⚠️ No devices found", fg='#f39c12')
                    self.root.after(0, lambda: messagebox.showwarning("Scan Results", "No devices found"))
                
            except Exception as e:
                self.status_label.config(text="❌ Scan failed", fg='#e74c3c')
                self.root.after(0, lambda: messagebox.showerror("Scan Error", str(e)))
            finally:
                self.scan_btn.config(state=tk.NORMAL, text="🔍 Scan for Devices")
        
        threading.Thread(target=scan_thread, daemon=True).start()
    
    def display_scan_results(self, devices):
        """Display scanned devices in the results frame"""
        # Clear previous results
        for widget in self.scan_results_frame.winfo_children():
            widget.destroy()
        
        if not devices:
            return
        
        tk.Label(
            self.scan_results_frame,
            text="Found Devices (click to copy MAC):",
            font=('Arial', 9, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        ).pack(anchor=tk.W, pady=2)
        
        # Create clickable buttons for each device
        for i, (mac, name) in enumerate(devices[:6]):  # Show max 6 devices
            btn = tk.Button(
                self.scan_results_frame,
                text=f"{name}\n{mac}",
                font=('Arial', 8),
                bg='#3d3d3d',
                fg='#ffffff',
                relief=tk.RAISED,
                borderwidth=1,
                command=lambda m=mac, n=name: self.select_scanned_device(m, n)
            )
            btn.pack(fill=tk.X, pady=2)
    
    def select_scanned_device(self, mac, name):
        """Handle selection of a scanned device"""
        # Find first empty MAC entry or ask user
        for i in range(1, 4):
            if not self.mac_entries[i].get().strip():
                self.mac_entries[i].delete(0, tk.END)
                self.mac_entries[i].insert(0, mac)
                self.status_label.config(text=f"✓ Added {name} to Device {i}", fg='#50c878')
                return
        
        # All slots filled, show dialog
        messagebox.showinfo("MAC Address Copied", 
                          f"Device: {name}\nMAC: {mac}\n\nAll slots are filled. Replace a MAC address manually if needed.")
    
    def connect_devices(self):
        """Connect to the ESP32 devices"""
        
        # Create manager
        self.manager = MultiDeviceManager()
        
        # Add devices from MAC entries
        for i in range(1, 4):
            mac = self.mac_entries[i].get().strip()
            if mac:
                self.manager.add_device(i, f"ESP32_HR_{i}", mac)
        
        if not self.manager.devices:
            messagebox.showwarning("Connection", "Please enter at least one MAC address")
            return
        
        self.status_label.config(text="🔗 Connecting...", fg='#f39c12')
        self.connect_btn.config(state=tk.DISABLED)
        
        def connect_thread():
            self.manager.connect_all()
            
            # Check if any connected
            connected = any(d.connected for d in self.manager.devices)
            
            if connected:
                self.manager.start_receiving()
                self.running = True
                self.status_label.config(text="🟢 Connected", fg='#50c878')
                self.disconnect_btn.config(state=tk.NORMAL)
                # Switch to overview tab after connection
                self.notebook.select(1)
            else:
                self.status_label.config(text="❌ Failed", fg='#e74c3c')
                self.connect_btn.config(state=tk.NORMAL)
                messagebox.showerror("Connection Error", "Failed to connect to any device")
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def disconnect_devices(self):
        """Disconnect from devices"""
        if self.manager:
            self.manager.stop()
            self.running = False
        
        self.status_label.config(text="⚪ Disconnected", fg='#888888')
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
    
    def update_display(self):
        """Update the display with current data"""
        
        if self.manager and self.running:
            all_data = self.manager.get_all_data()
            
            for device_id, device_info in all_data.items():
                if device_id in self.device_frames:
                    frame = self.device_frames[device_id]
                    data = device_info['data']
                    metrics = device_info['metrics']
                    connected = device_info['connected']
                    smoothed_hr = device_info.get('smoothed_hr')
                    smoothed_spo2 = device_info.get('smoothed_spo2')
                    
                    # Update current readings - use smoothed values for display
                    if connected and smoothed_hr is not None:
                        frame.hr_label.config(text=str(smoothed_hr))
                        
                        # Color based on HR range
                        if 60 <= smoothed_hr <= 100:
                            frame.hr_label.config(fg='#50c878')  # Green
                        elif 40 <= smoothed_hr < 60 or 100 < smoothed_hr <= 120:
                            frame.hr_label.config(fg='#f39c12')  # Orange
                        else:
                            frame.hr_label.config(fg='#e74c3c')  # Red
                    elif connected and data['hr'] is not None:
                        # Fallback to raw value if smoothed not available yet
                        hr_text = str(data['hr']) if data['hr_valid'] else "--"
                        frame.hr_label.config(text=hr_text)
                    
                    if connected and smoothed_spo2 is not None:
                        frame.spo2_label.config(text=f"{smoothed_spo2}%")
                    elif connected and data['spo2'] is not None:
                        # Fallback to raw value if smoothed not available yet
                        spo2_text = f"{data['spo2']}%" if data['spo2_valid'] else "--"
                        frame.spo2_label.config(text=spo2_text)
                    
                    # Update status
                    status = data['status']
                    if status == 'receiving':
                        frame.status_label.config(text="✓ Receiving", fg='#50c878')
                    elif status == 'NO_FINGER':
                        frame.status_label.config(text="⚠ No Finger", fg='#f39c12')
                    elif status == 'connected':
                        frame.status_label.config(text="Connected", fg='#4a90e2')
                    else:
                        frame.status_label.config(text=status, fg='#e74c3c')
                    
                    # Update metrics
                    if metrics['bpm'] is not None:
                        frame.metrics_labels['bpm'].config(text=f"{metrics['bpm']:.1f}")
                    if metrics['ipm'] is not None:
                        frame.metrics_labels['ipm'].config(text=f"{metrics['ipm']:.1f}")
                    if metrics['hrstd'] is not None:
                        frame.metrics_labels['hrstd'].config(text=f"{metrics['hrstd']:.2f}")
                    if metrics['rmssd'] is not None:
                        frame.metrics_labels['rmssd'].config(text=f"{metrics['rmssd']:.2f}")
                    
                    # Collect plot data - use smoothed values for cleaner graphs
                    if smoothed_hr is not None:
                        current_time = time.time()
                        self.plot_data[device_id]['time'].append(current_time)
                        self.plot_data[device_id]['hr'].append(smoothed_hr)
                        
                        # Keep only last max_plot_points
                        if len(self.plot_data[device_id]['time']) > self.max_plot_points:
                            self.plot_data[device_id]['time'].pop(0)
                            self.plot_data[device_id]['hr'].pop(0)
            
            # Update plot
            self.update_plot()
        
        # Schedule next update
        self.root.after(self.update_interval, self.update_display)
    
    def update_plot(self):
        """Update the heart rate plot"""
        self.ax.clear()
        
        current_time = time.time()
        colors = ['#e74c3c', '#50c878', '#4a90e2']
        device_names = ['Device 1', 'Device 2', 'Device 3']
        
        has_data = False
        for device_id in [1, 2, 3]:
            if len(self.plot_data[device_id]['time']) > 0:
                has_data = True
                times = np.array(self.plot_data[device_id]['time'])
                hrs = np.array(self.plot_data[device_id]['hr'])
                
                # Convert to seconds ago
                times_ago = current_time - times
                times_ago = times_ago[::-1]  # Reverse so most recent is at 0
                hrs = hrs[::-1]
                
                self.ax.plot(times_ago, hrs, 
                           color=colors[device_id - 1],
                           linewidth=2,
                           marker='o',
                           markersize=3,
                           label=device_names[device_id - 1],
                           alpha=0.9)
        
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_xlabel('Seconds Ago', color='white', fontsize=8)
        self.ax.set_ylabel('HR (bpm)', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=7)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        if has_data:
            self.ax.legend(loc='upper right', facecolor='#2d2d2d', 
                          edgecolor='white', labelcolor='white',
                          fontsize=7, framealpha=0.9)
        
        self.ax.set_xlim(self.max_plot_points, 0)
        self.ax.set_ylim(40, 140)
        
        # Adjust layout to fit compact space
        self.fig.tight_layout()
        
        try:
            self.canvas.draw()
        except:
            pass  # Ignore drawing errors during window close
    
    def on_closing(self):
        """Handle window close event"""
        if self.running:
            if messagebox.askokcancel("Quit", "Disconnect and quit?"):
                self.disconnect_devices()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = CompactHeartRateGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

