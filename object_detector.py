from object_detection_pb2 import BBoxes
import pickle

from ultralytics import YOLO  # docs: https://docs.ultralytics.com/modes/predict/


class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n')

    def detect(self, frame) -> BBoxes:
        output = []

        results = self.model(frame, verbose=False)

        result = results[0]

        labels_map = result.names

        boxes = result.boxes.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()

        score = max(conf) if len(conf) > 0 else 0

        for i, box in enumerate(boxes):
            class_label_id = box.cls[0]
            class_label = labels_map[class_label_id]
            r = box.xyxy[0].astype(int)
            output.append(
                [class_label, r[0], r[1], r[2], r[3], conf[i]]
            )
        res = BBoxes(data=pickle.dumps(output))
        return res, score
