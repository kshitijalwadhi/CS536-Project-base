import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse, DetectRequest, DetectResponse

from threading import Lock
from concurrent import futures

from utilities.constants import MAX_CAMERAS, IMG_SIZE, BW

from object_detector import ObjectDetector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import time
import random
import pickle
import cv2
import threading


class Server(object_detection_pb2_grpc.DetectorServicer):
    def __init__(self, detector=None):
        super(Server, self).__init__()
        self.detector = detector
        self.connected_clients = {}
        self.lock = Lock()
        self.current_load = 0
        self.prob_dropping = {}  # This is what will control our BW allocation
        self.past_scores = {}
        self.server_start_time = time.time()
        self.start_times = {}

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
            self.start_times[client_id] = time.time() - self.server_start_time
        print("Client with ID {} connected".format(client_id))
        return InitResponse(client_id=client_id)
    
    def plot_metrics(self):
        # Ensure this function is called periodically in the server's main loop or a separate thread.
        print("here")
        try:
            # Create a figure with two subplots
            plt.figure(figsize=(12, 6))

            # Plotting Object Detection Scores
            plt.subplot(1, 2, 1)
            for client_id, scores in self.past_scores.items():
                if scores:
                    relative_start_time = self.start_times.get(client_id, 0)
                    averaged_scores = [np.mean(scores[i:i+10]) for i in range(0, len(scores), 10)]
                    num_chunks = len(averaged_scores)
                    time_offsets = [relative_start_time + i for i in range(num_chunks)]
                    plt.plot(time_offsets, averaged_scores, label=f'Client {client_id}')
            plt.title('Object Detection Scores')
            plt.xlabel('Time')
            plt.ylabel('Score')
            plt.legend()

            # Plotting Probability of Dropped Packets
            plt.subplot(1, 2, 2)
            client_ids = list(self.prob_dropping.keys())
            probabilities = list(self.prob_dropping.values())
            plt.bar(client_ids, probabilities, color='blue')
            plt.title('Probability of Dropped Packets')
            plt.xlabel('Client ID')
            plt.ylabel('Probability')
            plt.xticks(client_ids)

            plt.tight_layout()
            
            # Save the plot to a file
            plt.savefig('./server_metrics.png')
            print("Plot saved to 'server_metrics.png'")
            plt.close()
        except Exception as e:
            print(f"Error occurred during plotting: {e}")

    def close_connection(self, request: CloseRequest, context):
        with self.lock:
            if request.client_id in self.connected_clients:
                del self.connected_clients[request.client_id]
                del self.prob_dropping[request.client_id]
        print("Client with ID {} disconnected".format(request.client_id))
        return CloseResponse(client_id=request.client_id)
    
    def calculate_weighted_utilization(self, client_id):
        # Calculate the average score for the client
        avg_score = sum(self.past_scores[client_id]) / len(self.past_scores[client_id]) if self.past_scores[client_id] else 0

        normalized_score = avg_score

        # Weighted utilization considers both utilization and object detection score
        return self.connected_clients[client_id]["utilization"] * (1 + normalized_score)

    def update_prob_dropping(self):
        # TODO: This function decides the probability of dropping a request, basically the BW allocation
        total_utilization = sum(self.calculate_weighted_utilization(client_id) 
                                for client_id in self.connected_clients)

        if total_utilization > BW:
            print("Current load is {}, exceeding BW: {}".format(total_utilization, BW))
            # sorted_clients = sorted(self.connected_clients.items(), key=lambda x: x[1]["utilization"], reverse=True)
            # top_client = sorted_clients[0][0]
            # self.prob_dropping[top_client] = 0.5
            # print("Client {} probability of dropping has been updated to: ".format(top_client), self.prob_dropping[top_client])
            weighted_utilization_proportions = {client_id: self.calculate_weighted_utilization(client_id) / total_utilization 
                                                for client_id in self.connected_clients}

            for client_id in self.connected_clients:
                inverse = 1 - weighted_utilization_proportions[client_id]
                self.prob_dropping[client_id] = min(1, inverse)

            # Log updated probabilities
            for client_id, prob in self.prob_dropping.items():
                print(f"Client {client_id} probability of dropping updated to: {prob}")
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

        print(f"Client {request.client_id} OD Score: {score}")

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
