import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse, DetectRequest, DetectResponse

from threading import Lock
from concurrent import futures

from utilities.constants import MAX_CAMERAS, IMG_SIZE, BW

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
        self.current_load = 0
        self.prob_dropping = {}  # This is what will control our BW allocation
        self.past_scores = {}

    def init_client(self, request: InitRequest, context):
        with self.lock:
            client_id = random.randint(1, MAX_CAMERAS)
            while client_id in self.connected_clients:
                client_id = random.randint(1, MAX_CAMERAS)
            self.connected_clients[client_id] = {
                "fps": 0,
                "size_each_frame": 0,
                "utilization": 0
            }
            self.prob_dropping[client_id] = 0
            self.past_scores[client_id] = []
        print("Client with ID {} connected".format(client_id))
        return InitResponse(client_id=client_id)

    def close_connection(self, request: CloseRequest, context):
        with self.lock:
            if request.client_id in self.connected_clients:
                del self.connected_clients[request.client_id]
                del self.prob_dropping[request.client_id]
        print("Client with ID {} disconnected".format(request.client_id))
        return CloseResponse(client_id=request.client_id)

    def update_prob_dropping(self):
        # TODO: This function decides the probability of dropping a request, basically the BW allocation
        if self.current_load > BW:
            print("Current load is {}, exceeding BW: {}".format(self.current_load, BW))
            sorted_clients = sorted(self.connected_clients.items(), key=lambda x: x[1]["utilization"], reverse=True)
            top_client = sorted_clients[0][0]
            self.prob_dropping[top_client] = 0.5
            print("Client {} probability of dropping has been updated to: ".format(top_client), self.prob_dropping[top_client])
        else:
            print("Current load is {}, within BW: {}".format(self.current_load, BW))
            sorted_prob_dropping = sorted(self.prob_dropping.items(), key=lambda x: x[1], reverse=True)
            top_client = sorted_prob_dropping[0][0]
            if self.prob_dropping[top_client] > 0:
                self.prob_dropping[top_client] = 0
                print("Client {} probability of dropping has been updated to: ".format(top_client), self.prob_dropping[top_client])

    def detect(self, request: DetectRequest, context):
        with self.lock:
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)
            self.connected_clients[request.client_id]["utilization"] = self.connected_clients[request.client_id]["size_each_frame"] * self.connected_clients[request.client_id]["fps"]

            self.current_load = 0
            for client_id in self.connected_clients:
                self.current_load += self.connected_clients[client_id]["utilization"] * (1-self.prob_dropping[client_id])

            self.update_prob_dropping()

        if random.random() < self.prob_dropping[request.client_id]:
            self.past_scores[request.client_id].append(0)
            with self.lock:
                self.current_load -= self.connected_clients[request.client_id]["utilization"] * self.prob_dropping[request.client_id]
            res = DetectResponse(
                client_id=request.client_id,
                sequence_number=request.sequence_number,
                req_dropped=True,
                bboxes=None
            )
            return res

        frame = pickle.loads(request.frame_data)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        bboxes, score = self.detector.detect(frame)

        self.past_scores[request.client_id].append(score)

        res = DetectResponse(
            client_id=request.client_id,
            sequence_number=request.sequence_number,
            req_dropped=False,
            bboxes=bboxes
        )

        with self.lock:
            self.current_load -= self.connected_clients[request.client_id]["utilization"] * (1-self.prob_dropping[request.client_id])

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
