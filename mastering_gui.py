# mastering_gui.py (v5.6 - Feature: MP3 Export)
# This version adds a checkbox to allow the user to create a high-quality,
# compressed MP3 file in addition to the archival WAV master.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from PIL import Image, ImageTk

from audio_mastering_engine import process_audio, EQ_PRESETS

try:
    from ttkthemes import ThemedTk
except ImportError:
    messagebox.showerror("Dependency Error", "The 'ttkthemes' library is not installed.\nPlease run: pip install ttkthemes Pillow")
    exit()

class MasteringApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="equilux")

        self.title("Audio Mastering Suite v5.6 (AI Enhanced)")
        self.geometry("800x1150") # Adjusted height
        self.configure(bg="#2b2b2b")
        self.photo_image = None

        style = ttk.Style()
        style.configure("TLabel", background="#2b2b2b", foreground="#cccccc", font=("Inter", 12))
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TButton", font=("Inter", 12, "bold"), padding=10)
        style.configure("Accent.TButton", background="#3c8eda", foreground="white")
        style.configure("TCheckbutton", background="#2b2b2b", foreground="#cccccc", font=("Inter", 12))
        style.map('TCheckbutton', background=[('active', '#3c3f41')], indicatorcolor=[('selected', '#82aaff'), ('!selected', '#555555')])
        style.configure("Value.TLabel", background="#2b2b2b", foreground="#82aaff", font=("Inter", 12, "bold"))
        style.configure("TScale", background="#2b2b2b")
        style.configure("Header.TLabel", font=("Inter", 14, "bold"), foreground="#82aaff")

        main_frame = ttk.Frame(self, padding="25 25 25 25")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=1)

        # --- VARIABLES ---
        self.input_file, self.output_file = tk.StringVar(), tk.StringVar()
        self.preset_var = tk.StringVar(value="None")
        self.analog_character, self.bass_boost, self.mid_cut = tk.DoubleVar(value=0), tk.DoubleVar(value=0), tk.DoubleVar(value=0)
        self.presence_boost, self.treble_boost = tk.DoubleVar(value=0), tk.DoubleVar(value=0)
        self.width, self.lufs = tk.DoubleVar(value=1.0), tk.DoubleVar(value=-14.0)
        self.use_multiband = tk.BooleanVar(value=False)
        self.low_thresh, self.low_ratio = tk.DoubleVar(value=-25.0), tk.DoubleVar(value=6.0)
        self.mid_thresh, self.mid_ratio = tk.DoubleVar(value=-20.0), tk.DoubleVar(value=3.0)
        self.high_thresh, self.high_ratio = tk.DoubleVar(value=-15.0), tk.DoubleVar(value=4.0)
        self.art_prompt = tk.StringVar()
        self.auto_generate_prompt = tk.BooleanVar(value=False)
        self.create_mp3 = tk.BooleanVar(value=True) # <-- New control for MP3

        # --- UI LAYOUT ---
        current_row = 0
        file_frame = ttk.Frame(main_frame); file_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15)); file_frame.grid_columnconfigure(1, weight=1); current_row += 1
        self.create_file_selector(file_frame, "Input File:", self.input_file, self.select_input_file, 0)
        self.create_file_selector(file_frame, "Output File:", self.output_file, self.select_output_file, 1)

        preset_frame = ttk.Frame(main_frame); preset_frame.grid(row=current_row, column=0, sticky="ew", pady=15); preset_frame.grid_columnconfigure(0, weight=1); current_row += 1
        preset_options = ["None"] + list(EQ_PRESETS.keys()); ttk.OptionMenu(preset_frame, self.preset_var, *preset_options, command=self.apply_preset).grid(row=0, column=0, sticky="ew")

        params_frame = ttk.Frame(main_frame); params_frame.grid(row=current_row, column=0, sticky="ew", pady=15); params_frame.grid_columnconfigure(1, weight=1); current_row += 1
        sliders = [("Analog Character (%)", self.analog_character, 0, 100), ("Bass (dB)", self.bass_boost, -6, 6), ("Mid Cut (dB)", self.mid_cut, 0, 6), ("Presence (dB)", self.presence_boost, -6, 6), ("Treble (dB)", self.treble_boost, -6, 6), ("Stereo Width", self.width, 0, 2), ("Target LUFS", self.lufs, -20, -6)]
        for i, (text, var, f, t) in enumerate(sliders): self.create_slider(params_frame, text, var, f, t, i)

        multiband_header_frame = ttk.Frame(main_frame); multiband_header_frame.grid(row=current_row, column=0, sticky="ew", pady=(15, 5)); current_row += 1
        ttk.Checkbutton(multiband_header_frame, text="Use Multiband Compressor", variable=self.use_multiband, command=self.toggle_multiband_frame).grid(row=0, column=0, sticky='w')
        
        self.multiband_controls_frame = ttk.Frame(main_frame); self.multiband_controls_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 15)); self.multiband_controls_frame.grid_columnconfigure(1, weight=1); current_row += 1
        mb_sliders = [("Low Thresh (dB)", self.low_thresh, -40, 0), ("Low Ratio", self.low_ratio, 1, 10), ("Mid Thresh (dB)", self.mid_thresh, -40, 0), ("Mid Ratio", self.mid_ratio, 1, 10), ("High Thresh (dB)", self.high_thresh, -40, 0), ("High Ratio", self.high_ratio, 1, 10)]
        for i, (text, var, f, t) in enumerate(mb_sliders): self.create_slider(self.multiband_controls_frame, text, var, f, t, i)
        
        art_header_frame = ttk.Frame(main_frame); art_header_frame.grid(row=current_row, column=0, sticky="ew", pady=(20, 5)); current_row += 1
        ttk.Label(art_header_frame, text="AI Cover Art & Final Output", style="Header.TLabel").grid(row=0, column=0, sticky='w')
        
        art_frame = ttk.Frame(main_frame); art_frame.grid(row=current_row, column=0, sticky="ew", pady=5); art_frame.grid_columnconfigure(1, weight=1); current_row += 1
        self.art_prompt_entry = ttk.Entry(art_frame, textvariable=self.art_prompt)
        self.art_prompt_entry.grid(row=0, column=1, sticky='ew', padx=10)
        ttk.Label(art_frame, text="Manual Art Prompt:").grid(row=0, column=0, sticky='w')
        
        auto_gen_check = ttk.Checkbutton(art_frame, text="Auto-generate prompt from audio analysis?", variable=self.auto_generate_prompt, command=self.toggle_art_prompt_entry)
        auto_gen_check.grid(row=1, column=1, sticky='w', padx=10, pady=5)

        # --- NEW MP3 CHECKBOX ---
        mp3_check = ttk.Checkbutton(art_frame, text="Also create a high-quality MP3 for listening?", variable=self.create_mp3)
        mp3_check.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        
        self.tag_label = ttk.Label(main_frame, text="Studio Notes: Ready for analysis.", relief=tk.SUNKEN, padding=5, wraplength=750)
        self.tag_label.grid(row=current_row, column=0, sticky="ew", pady=10); current_row += 1

        process_frame = ttk.Frame(main_frame); process_frame.grid(row=current_row, column=0, sticky="ewns", pady=15); process_frame.grid_columnconfigure(0, weight=1); current_row += 1
        self.process_button = ttk.Button(process_frame, text="Start Processing", style="Accent.TButton", command=self.start_processing_thread)
        self.process_button.grid(row=0, column=0, sticky="ew", pady=10)
        self.progress_bar = ttk.Progressbar(process_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=5)
        self.status_label = ttk.Label(process_frame, text="Ready.", relief=tk.SUNKEN, padding=5)
        self.status_label.grid(row=2, column=0, sticky="ew", pady=5)

        self.art_display_frame = ttk.Frame(main_frame, relief=tk.SUNKEN); self.art_display_frame.grid(row=current_row, column=0, sticky="nsew", pady=15); self.art_display_frame.grid_columnconfigure(0, weight=1); self.art_display_frame.grid_rowconfigure(0, weight=1); main_frame.grid_rowconfigure(current_row, weight=1); current_row += 1
        self.art_label = ttk.Label(self.art_display_frame, text="AI Art Will Appear Here", anchor="center"); self.art_label.grid(row=0, column=0, sticky="nsew")
        
        self.toggle_multiband_frame()

    def start_processing_thread(self):
        if not self.input_file.get() or not self.output_file.get():
            messagebox.showerror("Error", "Please select both an input and an output file.")
            return

        self.process_button.config(state=tk.DISABLED)
        self.status_label.config(text="Starting...")
        self.progress_bar["value"] = 0
        self.art_label.config(image=None, text="AI Art Will Appear Here")
        self.tag_label.config(text="Studio Notes: Beginning process...")

        settings = {
            "input_file": self.input_file.get(), "output_file": self.output_file.get(),
            "analog_character": self.analog_character.get(), "bass_boost": self.bass_boost.get(),
            "mid_cut": self.mid_cut.get(), "presence_boost": self.presence_boost.get(),
            "treble_boost": self.treble_boost.get(), "width": self.width.get(), "lufs": self.lufs.get(),
            "multiband": self.use_multiband.get(), "low_thresh": self.low_thresh.get(),
            "low_ratio": self.low_ratio.get(), "mid_thresh": self.mid_thresh.get(),
            "mid_ratio": self.mid_ratio.get(), "high_thresh": self.high_thresh.get(),
            "high_ratio": self.high_ratio.get(), "art_prompt": self.art_prompt.get(),
            "auto_generate_prompt": self.auto_generate_prompt.get(),
            "create_mp3": self.create_mp3.get() # <-- Pass the new setting
        }

        processing_thread = threading.Thread(
            target=process_audio,
            args=(settings, self.update_status, self.update_progress, self.update_art_display, self.update_tag_display)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
    # --- All other functions are unchanged ---
    def toggle_art_prompt_entry(self):
        if self.auto_generate_prompt.get():
            self.art_prompt_entry.config(state=tk.DISABLED)
            self.tag_label.config(text="Studio Notes: Awaiting audio analysis to generate prompt.")
        else:
            self.art_prompt_entry.config(state=tk.NORMAL)
            self.tag_label.config(text="Studio Notes: Using manual art prompt.")
            
    def update_tag_display(self, message): self.tag_label.config(text=f"Studio Notes: {message}")
    def create_file_selector(self, parent, label_text, string_var, command, row):
        ttk.Label(parent, text=label_text, width=15).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=string_var, state="readonly").grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(parent, text="Browse...", command=command).grid(row=row, column=2, sticky="e")
        parent.grid_columnconfigure(1, weight=1)
    def create_slider(self, parent, label_text, variable, from_, to, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=2, padx=5)
        scale = ttk.Scale(parent, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
        scale.grid(row=row, column=1, sticky="ew", padx=10)
        value_label = ttk.Label(parent, text=f"{variable.get():.1f}", width=8, style="Value.TLabel")
        value_label.grid(row=row, column=2, sticky="e", padx=5)
        parent.grid_columnconfigure(1, weight=1)
        variable.trace_add("write", lambda *args, var=variable, lbl=value_label: lbl.config(text=f"{var.get():.1f}"))
    def toggle_multiband_frame(self):
        if self.use_multiband.get(): self.multiband_controls_frame.grid()
        else: self.multiband_controls_frame.grid_remove()
    def apply_preset(self, preset_name):
        if preset_name == "None": self.bass_boost.set(0); self.mid_cut.set(0); self.presence_boost.set(0); self.treble_boost.set(0); return
        preset = EQ_PRESETS.get(preset_name)
        if preset: self.bass_boost.set(preset.get("bass_boost", 0)); self.mid_cut.set(preset.get("mid_cut", 0)); self.presence_boost.set(preset.get("presence_boost", 0)); self.treble_boost.set(preset.get("treble_boost", 0)); self.update_status(f"Loaded '{preset_name}' preset.")
    def select_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.aiff"), ("All files", "*.*")])
        if file_path: self.input_file.set(file_path); path, filename = os.path.split(file_path); name, ext = os.path.splitext(filename); self.output_file.set(os.path.join(path, f"{name}_mastered.wav"))
    def select_output_file(self):
        file_path = filedialog.asksaveasfilename(filetypes=[("WAV file", "*.wav")], defaultextension=".wav")
        if file_path: self.output_file.set(file_path)
    def update_status(self, message):
        self.status_label.config(text=message)
        if "Success:" in message or "Error:" in message or "Failed:" in message:
            self.process_button.config(state=tk.NORMAL)
            if "Success:" in message: messagebox.showinfo("Success", "Your audio file has been processed successfully!")
    def update_progress(self, current_step, total_steps):
        if total_steps > 0: self.progress_bar["maximum"] = total_steps; self.progress_bar["value"] = current_step
    def update_art_display(self, image_path):
        if not image_path: return
        try:
            img = Image.open(image_path)
            frame_width = self.art_display_frame.winfo_width() or 400; frame_height = self.art_display_frame.winfo_height() or 400
            img_ratio = img.width / img.height; frame_ratio = frame_width / frame_height
            if img_ratio > frame_ratio: new_width, new_height = frame_width, int(frame_width / img_ratio)
            else: new_height, new_width = frame_height, int(frame_height * img_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img)
            self.art_label.config(image=self.photo_image, text="")
        except Exception as e:
            self.art_label.config(text=f"Error displaying image:\n{e}"); logging.error(f"Failed to display image: {e}")

if __name__ == "__main__":
    app = MasteringApp()
    app.mainloop()

