import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("SPECTRA : Hyperspectral Surveillance")
root.geometry("1200x800")
root.configure(bg="#1e1e1e")

# ---------------- Grid config ----------------
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)

for i in range(20):
    root.grid_rowconfigure(i, weight=1)

# ---------------- ROW 1 : FULL-WIDTH HEADER ----------------
header = tk.Label(
    root,
    text="SPECTRA : HYPERSPECTRAL SURVEILLANCE",
    font=("Segoe UI", 16, "bold"),
    fg="white",
    bg="#2b2b2b",
    pady=12
)
header.grid(row=0, column=0, columnspan=2,
            sticky="nsew", padx=6, pady=6)

# ---------------- Helper ----------------
def stream_box(title, row_start, row_span, col=0):
    frame = ttk.LabelFrame(root, text=title)
    frame.grid(row=row_start, column=col, rowspan=row_span,
               sticky="nsew", padx=8, pady=6)

    canvas = tk.Canvas(frame, bg="black")
    canvas.pack(fill="both", expand=True)
    return canvas

# ---------------- Column 1 : Streams ----------------
rgb_canvas = stream_box("RGB Stream",       row_start=1,  row_span=5, col=0)
ir_canvas  = stream_box("Infrared Stream", row_start=6,  row_span=5, col=0)
th_canvas  = stream_box("Thermal Stream",  row_start=11, row_span=5, col=0)

# ---------------- Column 2 : Heatmap ----------------
heatmap_canvas = stream_box("Heatmap", row_start=1, row_span=5, col=1)

# ---------------- Detection Console ----------------
detect_frame = ttk.LabelFrame(root, text="Detection Console")
detect_frame.grid(row=6, column=1, rowspan=4,
                  sticky="nsew", padx=8, pady=6)

for item in ["Person", "Vehicle", "Background"]:
    tk.Label(
        detect_frame,
        text=item,
        font=("Segoe UI", 12),
        anchor="w"
    ).pack(fill="x", padx=10, pady=6)

# ---------------- Stats Table ----------------
stats_frame = ttk.LabelFrame(root, text="Stats")
stats_frame.grid(row=10, column=1, rowspan=6,
                 sticky="nsew", padx=8, pady=6)

for stat in ["Accuracy", "Threat Score", "Timestamp", "FPS"]:
    row = tk.Frame(stats_frame)
    row.pack(fill="x", padx=6, pady=4)

    tk.Label(row, text=stat, width=14, anchor="w").pack(side="left")
    tk.Label(row, text="--").pack(side="left")

# ---------------- Run ----------------
root.mainloop()
