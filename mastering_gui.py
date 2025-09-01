# mastering_gui.py (v2.4 - Local Desktop Version)
#
# This version includes a UI layout fix:
# - Removes the redundant and buggy section headers for a cleaner look.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

# This is the crucial link to our new engine.
from audio_mastering_engine import process_audio, EQ_PRESETS

# We use ttkthemes for a modern, dark look.
try:
    from ttkthemes import ThemedTk
except ImportError:
    messagebox.showerror("Dependency Error", "The 'ttkthemes' library is not installed.\nPlease run: pip install ttkthemes")
    exit()

class MasteringApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="equilux")

        self.title("Audio Mastering Suite v2.0")
        self.geometry("800x850") # Adjusted height
        self.configure(bg="#2b2b2b")

        # --- FONT & STYLE CONFIG ---
        style = ttk.Style()
        style.configure("TLabel", background="#2b2b2b", foreground="#cccccc", font=("Inter", 12))
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TButton", font=("Inter", 12, "bold"), padding=10)
        style.configure("Accent.TButton", background="#3c8eda", foreground="white")
        style.configure("TCheckbutton", background="#2b2b2b", foreground="#cccccc", font=("Inter", 12))
        style.map('TCheckbutton',
            background=[('active', '#3c3f41')],
            indicatorcolor=[('selected', '#82aaff'), ('!selected', '#555555')]
        )
        style.configure("Value.TLabel", background="#2b2b2b", foreground="#82aaff", font=("Inter", 12, "bold"))
        style.configure("TScale", background="#2b2b2b")

        # --- MAIN FRAME using GRID ---
        main_frame = ttk.Frame(self, padding="25 25 25 25")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=1)

        # --- VARIABLES ---
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.preset_var = tk.StringVar(value="None")
        self.analog_character = tk.DoubleVar(value=0)
        self.bass_boost = tk.DoubleVar(value=0)
        self.mid_cut = tk.DoubleVar(value=0)
        self.presence_boost = tk.DoubleVar(value=0)
        self.treble_boost = tk.DoubleVar(value=0)
        self.width = tk.DoubleVar(value=1.0)
        self.lufs = tk.DoubleVar(value=-14.0)
        self.use_multiband = tk.BooleanVar(value=False)
        self.low_thresh = tk.DoubleVar(value=-25.0)
        self.low_ratio = tk.DoubleVar(value=6.0)
        self.mid_thresh = tk.DoubleVar(value=-20.0)
        self.mid_ratio = tk.DoubleVar(value=3.0)
        self.high_thresh = tk.DoubleVar(value=-15.0)
        self.high_ratio = tk.DoubleVar(value=4.0)

        # --- UI LAYOUT using a pure GRID system ---
        current_row = 0
        
        # 1. File Selection Frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 20))
        file_frame.grid_columnconfigure(1, weight=1)
        current_row += 1
        self.create_file_selector(file_frame, "Input File:", self.input_file, self.select_input_file, 0)
        self.create_file_selector(file_frame, "Output File:", self.output_file, self.select_output_file, 1)

        # 2. Presets Frame
        preset_frame = ttk.Frame(main_frame)
        preset_frame.grid(row=current_row, column=0, sticky="ew", pady=20)
        preset_frame.grid_columnconfigure(0, weight=1)
        current_row += 1
        preset_options = ["None"] + list(EQ_PRESETS.keys())
        preset_menu = ttk.OptionMenu(preset_frame, self.preset_var, *preset_options, command=self.apply_preset)
        preset_menu.grid(row=0, column=0, sticky="ew")

        # 3. Main Parameters Frame
        params_frame = ttk.Frame(main_frame)
        params_frame.grid(row=current_row, column=0, sticky="ew", pady=20)
        params_frame.grid_columnconfigure(1, weight=1)
        current_row += 1
        self.create_slider(params_frame, "Analog Character (%)", self.analog_character, 0, 100, 0)
        self.create_slider(params_frame, "Bass (dB)", self.bass_boost, -6, 6, 1)
        self.create_slider(params_frame, "Mid Cut (dB)", self.mid_cut, 0, 6, 2)
        self.create_slider(params_frame, "Presence (dB)", self.presence_boost, -6, 6, 3)
        self.create_slider(params_frame, "Treble (dB)", self.treble_boost, -6, 6, 4)
        self.create_slider(params_frame, "Stereo Width", self.width, 0, 2, 5)
        self.create_slider(params_frame, "Target LUFS", self.lufs, -20, -6, 6)

        # 4. Multiband Compressor Frame
        multiband_header_frame = ttk.Frame(main_frame)
        multiband_header_frame.grid(row=current_row, column=0, sticky="ew", pady=(20, 5))
        current_row += 1
        ttk.Checkbutton(multiband_header_frame, text="Use Multiband Compressor", variable=self.use_multiband, command=self.toggle_multiband_frame).grid(row=0, column=0, sticky='w')
        
        self.multiband_controls_frame = ttk.Frame(main_frame)
        self.multiband_controls_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 20))
        self.multiband_controls_frame.grid_columnconfigure(1, weight=1)
        current_row += 1
        self.create_slider(self.multiband_controls_frame, "Low Thresh (dB)", self.low_thresh, -40, 0, 0)
        self.create_slider(self.multiband_controls_frame, "Low Ratio", self.low_ratio, 1, 10, 1)
        self.create_slider(self.multiband_controls_frame, "Mid Thresh (dB)", self.mid_thresh, -40, 0, 2)
        self.create_slider(self.multiband_controls_frame, "Mid Ratio", self.mid_ratio, 1, 10, 3)
        self.create_slider(self.multiband_controls_frame, "High Thresh (dB)", self.high_thresh, -40, 0, 4)
        self.create_slider(self.multiband_controls_frame, "High Ratio", self.high_ratio, 1, 10, 5)
        
        # 5. Process and Status Frame
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=current_row, column=0, sticky="ewns", pady=20)
        process_frame.grid_columnconfigure(0, weight=1)
        current_row += 1
        main_frame.grid_rowconfigure(current_row -1, weight=1) 
        
        self.process_button = ttk.Button(process_frame, text="Start Processing", style="Accent.TButton", command=self.start_processing_thread)
        self.process_button.grid(row=0, column=0, sticky="ew", pady=10)
        self.progress_bar = ttk.Progressbar(process_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=5)
        self.status_label = ttk.Label(process_frame, text="Ready.", relief=tk.SUNKEN, padding=5)
        self.status_label.grid(row=2, column=0, sticky="ew", pady=5)
        
        self.toggle_multiband_frame() 

    # --- UI HELPER METHODS ---
    def create_file_selector(self, parent, label_text, string_var, command, row):
        label = ttk.Label(parent, text=label_text, width=15)
        label.grid(row=row, column=0, sticky="w", pady=5)
        entry = ttk.Entry(parent, textvariable=string_var, state="readonly")
        entry.grid(row=row, column=1, sticky="ew", padx=5)
        button = ttk.Button(parent, text="Browse...", command=command)
        button.grid(row=row, column=2, sticky="e")
        parent.grid_columnconfigure(1, weight=1)

    def create_slider(self, parent, label_text, variable, from_, to, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", pady=2, padx=5)
        
        scale = ttk.Scale(parent, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
        scale.grid(row=row, column=1, sticky="ew", padx=10)
        
        value_label = ttk.Label(parent, text=f"{variable.get():.1f}", width=8, style="Value.TLabel")
        value_label.grid(row=row, column=2, sticky="e", padx=5)
        
        parent.grid_columnconfigure(1, weight=1)
        variable.trace_add("write", lambda *args, var=variable, lbl=value_label: lbl.config(text=f"{var.get():.1f}"))

    def toggle_multiband_frame(self):
        if self.use_multiband.get():
            self.multiband_controls_frame.grid()
        else:
            self.multiband_controls_frame.grid_remove()

    def apply_preset(self, preset_name):
        if preset_name == "None":
            self.bass_boost.set(0)
            self.mid_cut.set(0)
            self.presence_boost.set(0)
            self.treble_boost.set(0)
            return
        
        preset = EQ_PRESETS.get(preset_name)
        if preset:
            self.bass_boost.set(preset.get("bass_boost", 0))
            self.mid_cut.set(preset.get("mid_cut", 0))
            self.presence_boost.set(preset.get("presence_boost", 0))
            self.treble_boost.set(preset.get("treble_boost", 0))
            self.update_status(f"Loaded '{preset_name}' preset.")

    # --- FILE DIALOGS ---
    def select_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.aiff"), ("All files", "*.*")]
        )
        if file_path:
            self.input_file.set(file_path)
            path, filename = os.path.split(file_path)
            name, ext = os.path.splitext(filename)
            self.output_file.set(os.path.join(path, f"{name}_mastered{ext}"))

    def select_output_file(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Mastered File As...",
            filetypes=[("WAV file", "*.wav"), ("MP3 file", "*.mp3")],
            defaultextension=".wav"
        )
        if file_path:
            self.output_file.set(file_path)

    # --- PROCESSING LOGIC ---
    def start_processing_thread(self):
        if not self.input_file.get() or not self.output_file.get():
            messagebox.showerror("Error", "Please select both an input and an output file.")
            return

        self.process_button.config(state=tk.DISABLED)
        self.status_label.config(text="Starting...")
        self.progress_bar["value"] = 0
        
        settings = {
            "input_file": self.input_file.get(),
            "output_file": self.output_file.get(),
            "analog_character": self.analog_character.get(),
            "bass_boost": self.bass_boost.get(),
            "mid_cut": self.mid_cut.get(),
            "presence_boost": self.presence_boost.get(),
            "treble_boost": self.treble_boost.get(),
            "width": self.width.get(),
            "lufs": self.lufs.get(),
            "multiband": self.use_multiband.get(),
            "low_thresh": self.low_thresh.get(),
            "low_ratio": self.low_ratio.get(),
            "mid_thresh": self.mid_thresh.get(),
            "mid_ratio": self.mid_ratio.get(),
            "high_thresh": self.high_thresh.get(),
            "high_ratio": self.high_ratio.get(),
        }

        processing_thread = threading.Thread(
            target=process_audio,
            args=(settings, self.update_status, self.update_progress)
        )
        processing_thread.daemon = True
        processing_thread.start()

    # --- CALLBACKS FOR THE ENGINE ---
    def update_status(self, message):
        self.status_label.config(text=message)
        if "complete" in message.lower() or "error" in message.lower():
            self.process_button.config(state=tk.NORMAL)
            if "complete" in message.lower():
                messagebox.showinfo("Success", "Your audio file has been mastered successfully!")

    def update_progress(self, current_step, total_steps):
        self.progress_bar["maximum"] = total_steps
        self.progress_bar["value"] = current_step


if __name__ == "__main__":
    app = MasteringApp()
    app.mainloop()