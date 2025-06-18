# Person Detection and Counting with YOLOv8 and Supervision

This project implements a real-time person detection and counting system using YOLOv8 and the Supervision library. It can detect and track people across multiple predefined zones in a video stream, providing visual feedback and counting statistics.

## Features

- Real-time person detection using YOLOv8
- Multiple zone tracking with customizable polygons
- Visual annotations with bounding boxes and zone highlights
- Person counting per zone
- Support for video input/output processing
- Color-coded zones for easy visualization

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- Supervision
- CUDA-compatible GPU (recommended for real-time processing)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/person_detection_and_count_with-SUPERVISION-libraryandYOLOV8.git
cd person_detection_and_count_with-SUPERVISION-libraryandYOLOV8
```

2. Install the required dependencies:
```bash
pip install ultralytics supervision opencv-python numpy
```

3. Download the YOLOv8 model (will be downloaded automatically on first run):
```bash
# The model will be downloaded automatically when running the script
# Alternatively, you can manually download yolov8s.pt from the Ultralytics repository
```

## Usage

Run the person detection and counting script with the following command:

```bash
python person_count.py -i input_video.mp4 -o output_video.mp4
```

### Arguments

- `-i, --input`: Path to the input video file (required)
- `-o, --output`: Path to save the output video file (required)

## How It Works

1. The system uses YOLOv8 for person detection in each frame
2. Detected persons are filtered based on confidence threshold (>0.5)
3. The video is divided into multiple predefined zones using polygons
4. Each zone is color-coded and annotated with bounding boxes
5. The system tracks and counts people entering/leaving each zone
6. Results are visualized in real-time on the output video

## Project Structure

- `person_count.py`: Main script for person detection and counting
- `yolov8s.pt`: YOLOv8 model weights
- `YOO-V8_detect.ipynb`: Jupyter notebook with additional examples and experiments
- `demo2.mp4`: Sample input video
- `result.mp4`: Sample output video

## Customization

You can customize the following aspects of the system:

- Zone polygons: Modify the `polygons` list in the `CountObject` class
- Detection confidence threshold: Adjust the confidence value in the `process_frame` method
- Colors: Modify the `colors` list in the `CountObject` class
- Annotation styles: Adjust parameters in the `zone_annotators` and `box_annotators` initialization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision](https://github.com/roboflow/supervision)

