import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse, DetectRequest, DetectResponse

from threading import Lock
from concurrent import futures

from utilities.constants import MAX_CAMERAS, IMG_SIZE, BW, PAST_SCORE_N, MAX_TOTAL_FPS, MIN_THRESHOLD_EACH

from object_detector import ObjectDetector

import time
import random
import pickle
import cv2

import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD, LpMinimize
from scipy.optimize import linprog


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
            self.prob_dropping[client_id] = 0.0
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

        print("Current load is {}, BW: {}".format(self.current_load, BW))

        prob = LpProblem("BandwidthAllocation", LpMinimize)
        prob_vars = {client_id: LpVariable(f'prob_{client_id}', lowBound=0, upBound=0.9, cat='continuous')
                     for client_id in self.connected_clients}
        accuracy = {client_id: np.mean(self.past_scores[client_id][-PAST_SCORE_N:]) if self.past_scores[client_id] else 0
                    for client_id in self.connected_clients}

        print("Accuracy arr: ", accuracy)

        bw_arr = [self.connected_clients[client_id]["utilization"] * (1 - prob_vars[client_id]) for client_id in self.connected_clients]

        prob += lpSum(prob_vars), "Objective function"

        prob += lpSum(bw_arr) <= BW, "BW constraint"

        results = prob.solve(PULP_CBC_CMD(msg=0))
        print(LpStatus[results])
        if LpStatus[results] == 'Optimal':
            for client_id in self.connected_clients:
                if prob_vars[client_id].varValue is not None:
                    self.prob_dropping[client_id] = prob_vars[client_id].varValue

        for client_id in self.connected_clients:
            print("Client {} probability: ".format(client_id), self.prob_dropping[client_id])

    def update_prob_dropping_scipy(self):
        obj = [1.0 for client_id in range(len(self.connected_clients))]
        sum_util = np.sum([self.connected_clients[client_id]["utilization"] for client_id in self.connected_clients])
        rhs_ineq = [BW - sum_util]
        lhs_ineq = [[(-1) * self.connected_clients[client_id]["utilization"] for client_id in self.connected_clients]]

        bnd = [(0.001, 0.9) for _ in range(len(self.connected_clients))]

        print("LHS: ", lhs_ineq)
        print("RHS: ", rhs_ineq)

        opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="revised simplex")

        if opt.success:
            for i, client_id in enumerate(self.connected_clients):
                self.prob_dropping[client_id] = opt.x[i]

        for client_id in self.connected_clients:
            print("Client {} probability: ".format(client_id), self.prob_dropping[client_id])

    def detect(self, request: DetectRequest, context):
        with self.lock:
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)
            self.connected_clients[request.client_id]["utilization"] = self.connected_clients[request.client_id]["size_each_frame"] * self.connected_clients[request.client_id]["fps"]

            self.current_load = 0
            for client_id in self.connected_clients:
                print(self.prob_dropping)
                self.current_load += self.connected_clients[client_id]["utilization"] * (1-self.prob_dropping[client_id])

            try:
                self.update_prob_dropping_scipy()
            except Exception as e:
                print(e)

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
