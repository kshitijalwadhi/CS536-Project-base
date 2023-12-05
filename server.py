import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse, DetectRequest, DetectResponse

from threading import Lock
from concurrent import futures

from utilities.constants import MAX_CAMERAS, IMG_SIZE

from object_detector import ObjectDetector

import time
import random
import pickle
import cv2


class Server(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None):
        super(Server, self).__init__()
        self.detector = detector
        self.connected_clients = {}
        self.lock = Lock()

    def init_client(self, request: InitRequest, context):
        with self.lock:
            client_id = random.randint(1, MAX_CAMERAS)
            while client_id in self.connected_clients:
                client_id = random.randint(1, MAX_CAMERAS)
            self.connected_clients[client_id] = {}
        print("Client with ID {} connected".format(client_id))
        return InitResponse(client_id=client_id)

    def close_connection(self, request: CloseRequest, context):
        with self.lock:
            if request.client_id in self.connected_clients:
                del self.connected_clients[request.client_id]
        print("Client with ID {} disconnected".format(request.client_id))
        return CloseResponse(client_id=request.client_id)

    def detect(self, request: DetectRequest, context):
        with self.lock:
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)

        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        bboxes, score = self.detector.detect(frame)

        res = DetectResponse(
            client_id=request.client_id,
            sequence_number=request.sequence_number,
            req_dropped=False,
            bboxes=bboxes
        )

        return res


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    object_detection_pb2_grpc.add_DetectorServicer_to_server(Server(detector=ObjectDetector()), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
