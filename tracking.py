import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def main(video_path, class_name):
    """
    Process a video with YOLO and filter by class.
    """
    model = YOLO(r"weights/yolov8m.pt")
    names = model.model.names

    # Check if the class name is valid
    if class_name not in names.values():
        print(f"Error: Class '{class_name}' not found in the YOLO model.")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video writer
    result = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    class_dict = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection and tracking
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Filter classes
            filtered_boxes = []
            for box, cls, track_id, conf in zip(boxes, clss, track_ids, confs):
                # Check if the object has been detected before
                if track_id in class_dict:
                    # Keep the initial class of the object
                    cls = class_dict[track_id]
                else:
                    # Update the class_dict with the initial class of the object
                    class_dict[track_id] = int(cls)
                if names[int(cls)] == class_name:
                    filtered_boxes.append((box, cls, track_id, conf))

            # Draw a black space around the areas outside of the detected objects
            mask = np.zeros_like(frame)
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box[0][:4])
                mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            frame[:] = 0
            frame[mask != 0] = mask[mask != 0]

            # Draw annotations on the original frame
            annotator = Annotator(frame, line_width=3)
            for box, cls, track_id, conf in filtered_boxes:
                annotator.box_label(box, color=colors(int(cls), True), label=f"id {track_id}, {names[int(cls)]} {conf:.2}")

            annotated_frame = annotator.result()

            # Write the annotated frame to the output video
            result.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    result.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with YOLO and filter by class")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--selected_class", type=str, help="Class to filter by (e.g. 'cat')")
    args = parser.parse_args()

    main(args.video_path, args.selected_class)