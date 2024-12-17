import cv2
import torch
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import threading
import requests
import uuid
import yt_dlp

class YOLOHumanTracker:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)
        self.video_path = None
        self.running = False
        self.cap = None
        self.trackers = {}
        self.next_id = 1
        self.frame_skip = 2

    def load_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        if self.video_path:
            threading.Thread(target=self.process_video).start()

    def load_youtube_video(self):
        url = simpledialog.askstring("YouTube URL", "Enter YouTube Video URL:")
        if url:
            try:
                print("Fetching YouTube live stream...")
                ydl_opts = {'format': 'best'}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=False)
                    self.video_path = info_dict['url']
                print(f"Streaming URL: {self.video_path}")
                threading.Thread(target=self.process_video).start()
            except Exception as e:
                print(f"Error fetching YouTube live stream: {e}")

    def assign_unique_ids(self, results):
        current_trackers = {}
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = detection.tolist()
            if int(cls) == 0:
                bbox = (int(x1), int(y1), int(x2), int(y2))
                id_found = None
                for track_id, track_bbox in self.trackers.items():
                    if self.calculate_iou(track_bbox, bbox) > 0.5:
                        id_found = track_id
                        break
                if id_found is None:
                    id_found = str(uuid.uuid4())
                current_trackers[id_found] = bbox
        self.trackers = current_trackers

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        frame_count = 0

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            frame = cv2.resize(frame, (1280, 720))

            results = self.model(frame)
            self.assign_unique_ids(results)

            for track_id, bbox in self.trackers.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID: {track_id[:8]}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            self.panel.imgtk = frame_tk
            self.panel.configure(image=frame_tk)

        self.cap.release()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def create_gui(self):
        root = tk.Tk()
        root.title("YOLO Human Tracker")

        root.geometry("1300x800")
        root.resizable(True, True)

        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        load_button = tk.Button(button_frame, text="Load Video", command=self.load_video)
        load_button.pack(side=tk.LEFT, padx=5, pady=5)

        youtube_button = tk.Button(button_frame, text="Load YouTube", command=self.load_youtube_video)
        youtube_button.pack(side=tk.LEFT, padx=5, pady=5)

        stop_button = tk.Button(button_frame, text="Stop", command=self.stop_video)
        stop_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.panel = tk.Label(root, width=1280, height=720)
        self.panel.pack(padx=10, pady=10)

        root.protocol("WM_DELETE_WINDOW", self.stop_video)
        root.mainloop()

if __name__ == "__main__":
    tracker = YOLOHumanTracker()
    tracker.create_gui()
