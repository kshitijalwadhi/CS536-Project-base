from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import numpy as np
import threading
import matplotlib.pyplot as plt
import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse, DetectRequest, DetectResponse

from threading import Lock
from concurrent import futures

from utilities.constants import MAX_CAMERAS, IMG_SIZE, BW, PAST_SCORE_N, MIN_THRESHOLD_EACH

from object_detector import ObjectDetector

import time
import random
import pickle
import cv2

import matplotlib
matplotlib.use('Agg')


USE_LP_OPTIMIZATION = False

ADJUSTED_THRESHOLD = MIN_THRESHOLD_EACH


class Server(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None):
        super(Server, self).__init__()
        self.detector = detector
        self.connected_clients = {}
        self.connected_clients_plotting = {}
        self.lock = Lock()
        self.current_load = 0
        self.prob_dropping = {}  # This is what will control our BW allocation
        self.probs = {}
        self.past_scores = {}
        self.accuracies = {}
        self.od_scores = {}
        self.verbose = 1
        self.total_bandwidth = []
        self.bandwidths = {}
        self.server_start_time = time.time()
        self.start_times = {}
        self.client_times = {}
        self.server_times = []

    def init_client(self, request: InitRequest, context):
        with self.lock:
            client_id = random.randint(1, MAX_CAMERAS)
            while client_id in self.connected_clients:
                client_id = random.randint(1, MAX_CAMERAS)
            self.connected_clients[client_id] = {
                "fps": 1,
                "size_each_frame": 1024,
                "utilization": 1024
            }
            self.connected_clients_plotting[client_id] = {
                "fps": 1,
                "size_each_frame": 1024,
                "utilization": 1024
            }
            self.prob_dropping[client_id] = 0
            self.past_scores[client_id] = []
            self.past_scores[client_id].append(0)
            self.od_scores[client_id] = []
            self.od_scores[client_id].append(1)
            self.accuracies[client_id] = []
            self.accuracies[client_id].append(1)
            self.bandwidths[client_id] = []
            self.bandwidths[client_id].append(0)
            self.probs[client_id] = []
            self.probs[client_id].append(0)
            self.client_times[client_id] = []
            self.client_times[client_id].append(time.time() - self.server_start_time)
            self.start_times[client_id] = time.time() - self.server_start_time
            self.server_times.append(time.time() - self.server_start_time)
            self.total_bandwidth.append(0)

        print("Client with ID {} connected".format(client_id))
        return InitResponse(client_id=client_id)

    def close_connection(self, request: CloseRequest, context):
        with self.lock:
            if request.client_id in self.connected_clients:
                del self.connected_clients[request.client_id]
                del self.prob_dropping[request.client_id]
        print("Client with ID {} disconnected".format(request.client_id))
        return CloseResponse(client_id=request.client_id)

    def plot_metrics(self):
        # Create a figure with two subplots
        fig = plt.figure(figsize=(12, 6))

        # Plotting Object Detection Scores
        plt.subplot(3, 2, 1)
        total_bandwidth = self.total_bandwidth
        averaged_bw = [np.mean(total_bandwidth[i:i+5]) for i in range(0, len(total_bandwidth), 5)]
        times = [np.mean(self.server_times[i:i+5]) for i in range(0, len(self.server_times), 5)]

        plt.plot(times, averaged_bw)
        plt.title('Total Bandwidth over Time')
        plt.xlabel('Time')
        plt.ylabel('Total Bandwidth')

        # Plotting Accuracy of Each Client
        plt.subplot(3, 2, 2)
        for client_id, acc in self.accuracies.items():
            relative_start_time = self.start_times.get(client_id, 0)
            averaged_acc = [np.mean(acc[i:i+5]) for i in range(0, len(acc), 5)]
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            num_chunks = len(averaged_acc)
            time_offsets = [relative_start_time + i for i in range(5, num_chunks+5)]
            fps = self.connected_clients_plotting[client_id]['fps']
            plt.plot(times, averaged_acc, label=f'Client FPS: {fps}')

        plt.title('Accuracy of Each Client')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')

        # Plotting Bandwidth of Each Client
        plt.subplot(3, 2, 3)
        for client_id, bw in self.bandwidths.items():
            relative_start_time = self.start_times.get(client_id, 0)
            averaged_bw = [np.mean(bw[i:i+5]) for i in range(0, len(bw), 5)]
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            num_chunks = len(averaged_bw)
            time_offsets = [relative_start_time + i for i in range(5, num_chunks+5)]
            fps = self.connected_clients_plotting[client_id]['fps']
            plt.plot(times, averaged_bw, label=f'Client FPS: {fps}')

        plt.title('Bandwidth of Each Client')
        plt.xlabel('Time')
        plt.ylabel('Bandwidth')

        plt.subplot(3, 2, 4)
        for client_id, score in self.od_scores.items():
            relative_start_time = self.start_times.get(client_id, 0)
            averaged_scores = [np.mean(score[i:i+5]) for i in range(0, len(score), 5)]
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            num_chunks = len(averaged_scores)
            time_offsets = [relative_start_time + i for i in range(5, num_chunks+5)]
            fps = self.connected_clients_plotting[client_id]['fps']
            plt.plot(times, averaged_scores, label=f'Client FPS: {fps}')

        plt.title('Object Detection Scores of Each Client')
        plt.xlabel('Time')
        plt.ylabel('Score')

        plt.subplot(3, 2, 5)
        for client_id, prob in self.probs.items():
            relative_start_time = self.start_times.get(client_id, 0)
            averaged_prob = [np.mean(prob[i:i+5]) for i in range(0, len(prob), 5)]
            times = [np.mean(self.client_times[client_id][i:i+5]) for i in range(0, len(self.client_times[client_id]), 5)]
            num_chunks = len(averaged_prob)
            time_offsets = [relative_start_time + i for i in range(num_chunks)]
            fps = self.connected_clients_plotting[client_id]['fps']
            plt.plot(time_offsets, averaged_prob, label=f'Client FPS: {fps}')

        plt.title('Probability of Dropping a Packet')
        plt.xlabel('Time')
        plt.ylabel('Probability')

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('./server_metrics.png')
        print("Plot saved to 'server_metrics.png'")
        plt.close()
        

    def update_prob_dropping(self):

        print("Current load is {}, BW: {}".format(self.current_load, BW))

        prob = LpProblem("BandwidthAllocation", LpMaximize)
        fps_vars = {client_id: LpVariable(f'fps_{client_id}', lowBound=1, upBound=100, cat='Continuous')
                    for client_id in self.connected_clients}
        requested_fps = {client_id: client['fps'] for client_id, client in self.connected_clients.items()}
        accuracy = {client_id: np.mean(self.past_scores[client_id][-PAST_SCORE_N:]) if self.past_scores[client_id] else 0
                    for client_id in self.connected_clients}

        for client_id in accuracy.keys():
            self.od_scores[client_id].append(accuracy[client_id])
            self.accuracies[client_id].append(accuracy[client_id]*(1-self.prob_dropping[client_id]))
            self.client_times[client_id].append(time.time() - self.server_start_time)

        # add feature to drop clients with accuracy or performance below a certain number (min bound = accu_thresh*1/request_fps)
        for client_id in requested_fps:
            if self.verbose:
                print("id: ", client_id, " fps: ", requested_fps[client_id], " accuracy: ", accuracy[client_id])

        # objective
        prob += lpSum([fps_vars[client_id] * 1 / requested_fps[client_id]
                       for client_id in self.connected_clients])

        # BW constraint
        prob += lpSum([fps_vars[client_id] * self.connected_clients[client_id]['size_each_frame']
                       for client_id in self.connected_clients]) <= BW

        # #total fps capped
        # prob += lpSum([fps_vars[client_id] for client_id in self.connected_clients]) <= MAX_TOTAL_FPS

        ADJUSTED_THRESHOLD = MIN_THRESHOLD_EACH * np.exp(-len(self.connected_clients)/10)

        # #each fps capped
        for client_id in self.connected_clients:
            prob += fps_vars[client_id] <= self.connected_clients[client_id]['fps']

        # each performance min capped
        for client_id in self.connected_clients:
            prob += fps_vars[client_id] * accuracy[client_id] / requested_fps[client_id] >= ADJUSTED_THRESHOLD

        results = prob.solve(PULP_CBC_CMD(msg=0))
        # if self.verbose: print(prob)
        if LpStatus[results] == 'Optimal':
            for client_id in self.connected_clients:
                if fps_vars[client_id].varValue is not None:
                    self.prob_dropping[client_id] = 1.0 - fps_vars[client_id].varValue / requested_fps[client_id]
                    print("Client {} probability of dropping has been updated to: ".format(client_id), self.prob_dropping[client_id])

    def update_prob_dropping_simple(self):
        total_utilization = 0
        accuracy = {client_id: np.mean(self.past_scores[client_id][-PAST_SCORE_N:]) if self.past_scores[client_id] else 0
                    for client_id in self.connected_clients}
        for client_id in self.connected_clients:
            b_i = max(1-accuracy[client_id], MIN_THRESHOLD_EACH)
            total_utilization += self.connected_clients[client_id]["utilization"] * b_i
            self.client_times[client_id].append(time.time() - self.server_start_time)

        for client_id in self.connected_clients:
            u_i = self.connected_clients[client_id]["utilization"]
            b_i = max(1-accuracy[client_id], MIN_THRESHOLD_EACH)
            bw_i = (u_i * b_i) / total_utilization * BW
            p_i = max(0, 1 - bw_i/u_i)
            self.prob_dropping[client_id] = p_i

        for client_id in accuracy.keys():
            self.od_scores[client_id].append(accuracy[client_id])
            self.accuracies[client_id].append(accuracy[client_id]*(1-self.prob_dropping[client_id]))

        for client_id, prob in self.prob_dropping.items():
            print("Client: {client_id}, fps: {fps}, prob: {prob}".format(client_id=client_id, fps=self.connected_clients[client_id]["fps"], prob=prob))

    def detect(self, request: DetectRequest, context):
        with self.lock:
            self.connected_clients[request.client_id]["fps"] = request.fps
            self.connected_clients[request.client_id]["size_each_frame"] = len(request.frame_data)
            self.connected_clients[request.client_id]["utilization"] = self.connected_clients[request.client_id]["size_each_frame"] * self.connected_clients[request.client_id]["fps"]

            self.connected_clients_plotting[request.client_id]["fps"] = request.fps
            self.connected_clients_plotting[request.client_id]["size_each_frame"] = len(request.frame_data)
            self.connected_clients_plotting[request.client_id]["utilization"] = self.connected_clients_plotting[request.client_id]["size_each_frame"] * self.connected_clients_plotting[request.client_id]["fps"]

            self.current_load = 0
            for client_id in self.connected_clients:
                load = self.connected_clients[client_id]["utilization"] * (1-self.prob_dropping[client_id])
                self.current_load += load
                self.bandwidths[client_id].append(load)
                self.probs[client_id].append(self.prob_dropping[client_id])

            if USE_LP_OPTIMIZATION:
                self.update_prob_dropping()
            else:
                self.update_prob_dropping_simple()
            
            self.current_load = 0
            for client_id in self.connected_clients:
                load = self.connected_clients[client_id]["utilization"] * (1-self.prob_dropping[client_id])
                self.current_load += load
            self.total_bandwidth.append(self.current_load)
            self.server_times.append(time.time() - self.server_start_time)

        if random.random() < self.prob_dropping[request.client_id]:
            if not USE_LP_OPTIMIZATION:
                self.past_scores[request.client_id].append(0)
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

        return res


def start_plotting_thread(server_instance, interval=5):
    def plot():
        while not stop_plotting_thread.is_set():
            time.sleep(interval)
            server_instance.plot_metrics()

    stop_plotting_thread = threading.Event()
    plotting_thread = threading.Thread(target=plot)
    plotting_thread.start()
    return stop_plotting_thread, plotting_thread


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    od_server = Server(detector=ObjectDetector())
    object_detection_pb2_grpc.add_DetectorServicer_to_server(od_server, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    stop_event, plotting_thread = start_plotting_thread(od_server)
    print("Server started at port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        stop_event.set()  # Signal the plotting thread to stop
        plotting_thread.join()  # Wait for the plotting thread to finish
        server.stop(0)
