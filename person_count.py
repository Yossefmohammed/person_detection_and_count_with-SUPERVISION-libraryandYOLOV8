import numpy as np
import supervision as sv
from ultralytics import YOLO
import argparse

# Argument parser for input/output video paths
parser = argparse.ArgumentParser(
    prog='yolov8',
    description='This program helps detect and count people in polygon regions',
    epilog='Developed using Ultralytics and Supervision'
)
parser.add_argument('-i', '--input', required=True, help="Input video file path")
parser.add_argument('-o', '--output', required=True, help="Output video file path")
args = parser.parse_args()


class CountObject:
    def __init__(self, input_video_path, output_video_path) -> None:
        self.model = YOLO('yolov8s.pt')  # Automatically downloads if not found

        # Color palette for zones
        self.colors = sv.ColorPalette(colors=[
            sv.Color.from_hex("#FF0000"),  # Red
            sv.Color.from_hex("#00FF00"),  # Green
            sv.Color.from_hex("#0000FF"),  # Blue
            sv.Color.from_hex("#FFFF00"),  # Yellow
            sv.Color.from_hex("#FF00FF"),  # Magenta
            sv.Color.from_hex("#00FFFF"),  # Cyan
            sv.Color.from_hex("#FFA500")   # Orange
        ])

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # Predefined polygon regions
        self.polygons = [
            np.array([[540, 985], [1620, 985], [2160, 1920], [1620, 2855], [540, 2855], [0, 1920]], np.int32),
            np.array([[0, 1920], [540, 985], [0, 0]], np.int32),
            np.array([[1620, 985], [2160, 1920], [2160, 0]], np.int32),
            np.array([[540, 985], [0, 0], [2160, 0], [1620, 985]], np.int32),
            np.array([[0, 1920], [0, 3840], [540, 2855]], np.int32),
            np.array([[2160, 1920], [1620, 2855], [2160, 3840]], np.int32),
            np.array([[1620, 2855], [540, 2855], [0, 3840], [2160, 3840]], np.int32)
        ]

        self.video_info = sv.VideoInfo.from_video_path(input_video_path)

        # ✅ FIXED: Removed 'frame_resolution' argument
        self.zones = [
            sv.PolygonZone(polygon=polygon)
            for polygon in self.polygons
        ]

        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=self.colors.by_idx(index),
                thickness=6,
                text_thickness=8,
                text_scale=4
            )
            for index, zone in enumerate(self.zones)
        ]

        self.box_annotators = [
            sv.BoxAnnotator(color=self.colors.by_idx(index), thickness=4)
            for index in range(len(self.polygons))
        ]

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter for persons only
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]

            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)

        return frame

    def process_video(self):
        sv.process_video(
            source_path=self.input_video_path,
            target_path=self.output_video_path,
            callback=self.process_frame
        )


if __name__ == "__main__":
    obj = CountObject(args.input, args.output)
    obj.process_video()
