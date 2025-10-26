import threading, io
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog

# ---------- Genre Prediction ----------
def predict_genre(features):
    c, r, b, ct = features
    if c < 1500 and r < 3000: return "Classical"
    if 1500 <= c < 2500 and ct < 25: return "Jazz"
    if 2200 <= c < 4200 and 4000 <= r < 7000 and ct < 22 and b > 1500: return "Phonk"
    if c >= 3500 and ct >= 25 and r >= 5000: return "Hip-Hop / Rap"
    if c >= 5200 or r > 7000: return "Rock"
    if 2500 <= c < 3800 and r < 6000: return "Pop"
    return "Pop"

# ---------- Audio Feature Extraction ----------
def analyze_audio_file(path, duration=10):
    y, sr = librosa.load(path, sr=None, duration=duration)
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(S=D, sr=sr))
    return sr, D, [centroid, rolloff, bandwidth, contrast]

# ---------- GUI ----------
class SpotifyStyleGUI:
    def __init__(self, root):
        self.root = root
        self.audio_path = None

        # Spotify-like dark theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # Window title
        self.root.title("🎧 Music Genre Identifier (STFT)")
        self.root.configure(bg="#121212")

        # Main container
        main_frame = ctk.CTkFrame(root, corner_radius=20, fg_color="#121212")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title and subtitle
        title = ctk.CTkLabel(main_frame, text="🎵 Music Genre Identifier",
                             font=("Garamond", 30, "bold"), text_color="#1DB954")
        title.pack(pady=(15, 5))

        subtitle = ctk.CTkLabel(main_frame, text="Using Short-Time Fourier Transform (STFT)",
                                font=("Trebuchet MS", 14), text_color="#B3B3B3")
        subtitle.pack(pady=(0, 15))

        # Buttons frame
        btn_frame = ctk.CTkFrame(main_frame, fg_color="#181818", corner_radius=15)
        btn_frame.pack(pady=15)

        self.upload_btn = ctk.CTkButton(btn_frame, text="Upload Audio", width=170, height=45,
                                        corner_radius=25, fg_color="#1DB954", hover_color="#909090",
                                        font=("Trebuchet MS", 18, "bold"), command=self.load_audio)
        self.upload_btn.grid(row=0, column=0, padx=10, pady=10)

        self.analyze_btn = ctk.CTkButton(btn_frame, text="Analyze Audio", width=170, height=45,
                                         corner_radius=25, fg_color="#1DB954", hover_color="#909090",
                                         font=("Trebuchet MS", 18, "bold"), command=self.start_thread)
        self.analyze_btn.grid(row=0, column=1, padx=10, pady=10)

        # Status and result labels
        self.status_label = ctk.CTkLabel(main_frame, text="Status: Waiting for audio...",
                                         font=("Trebuchet MS", 14), text_color="#CCCCCC")
        self.status_label.pack(pady=6)

        self.result_label = ctk.CTkLabel(main_frame, text="Predicted Genre: -",
                                         font=("Trebuchet MS", 20, "bold"), text_color="#1DB954")
        self.result_label.pack(pady=6)

        # Filename display label
        self.filename_label = ctk.CTkLabel(main_frame, text="File: -",
                                           font=("Trebuchet MS", 14), text_color="#B3B3B3")
        self.filename_label.pack(pady=(0, 10))

        # Spectrogram area
        self.spectrogram_frame = ctk.CTkFrame(main_frame, fg_color="#181818", corner_radius=20)
        self.spectrogram_frame.pack(pady=15)
        self.spectrogram_label = ctk.CTkLabel(self.spectrogram_frame,
                                              text="Spectrogram will appear here",
                                              font=("Trebuchet MS", 13),
                                              text_color="#B3B3B3", width=820, height=340)
        self.spectrogram_label.pack(pady=10, padx=10)

    # ---------- Load + Analyze ----------
    def load_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")])
        if path:
            self.audio_path = path
            fname = path.split("/")[-1]
            self.status_label.configure(text=f"Loaded: {fname}", text_color="#1DB954",
                                        font=("Trebuchet MS", 15, "bold"))
            self.result_label.configure(text="Predicted Genre: -", text_color="#1DB954")
            self.filename_label.configure(text=f"File: {fname}", text_color="#B3B3B3")
            self.spectrogram_label.configure(image=None, text="Spectrogram will appear here")

    def start_thread(self):
        if not self.audio_path:
            self.status_label.configure(text="⚠️ Please upload an audio file.", text_color="red")
            return
        self.upload_btn.configure(state="disabled")
        self.analyze_btn.configure(state="disabled")
        self.status_label.configure(text="Analyzing...", text_color="#B3B3B3")
        threading.Thread(target=self.analyze_audio, daemon=True).start()

    def analyze_audio(self):
        try:
            fname = self.audio_path.split("/")[-1]
            sr, D, features = analyze_audio_file(self.audio_path)
            genre = predict_genre(features)

            # Update UI after analysis
            self.root.after(0, lambda: self.result_label.configure(
                text=f"🎧 Predicted Genre: {genre}", text_color="#1DB954"))
            self.root.after(0, lambda: self.status_label.configure(
                text="Status: Analysis complete.", text_color="#1DB954"))
            self.root.after(0, lambda: self.filename_label.configure(
                text=f"Analyzed File: {fname}", text_color="#B3B3B3"))

            # Generate spectrogram
            fig, ax = plt.subplots(figsize=(8, 3.5))
            librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                     sr=sr, hop_length=256, x_axis='time',
                                     y_axis='log', cmap='inferno', ax=ax)
            ax.set(title="Spectrogram (STFT)")
            plt.tight_layout()

            # Save plot to memory and display
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            img = Image.open(buf).resize((820, 340))
            imgtk = ImageTk.PhotoImage(img)

            def update_ui():
                self.spectrogram_label.configure(image=imgtk, text="")
                self.spectrogram_label.image = imgtk
                self.upload_btn.configure(state="normal")
                self.analyze_btn.configure(state="normal")

            self.root.after(0, update_ui)

        except Exception as e:
            self.root.after(0, lambda: self.status_label.configure(text=f"Error: {e}", text_color="red"))
            self.root.after(0, lambda: self.upload_btn.configure(state="normal"))
            self.root.after(0, lambda: self.analyze_btn.configure(state="normal"))

# ---------- Main ----------
if __name__ == "__main__":
    root = ctk.CTk()
    app_width, app_height = 950, 745

    # Center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (app_width / 2))
    y = int((screen_height / 2) - (app_height / 2))
    root.geometry(f"{app_width}x{app_height}+{x}+{y}")

    SpotifyStyleGUI(root)
    root.mainloop()
