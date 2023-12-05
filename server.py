import grpc

import object_detection_pb2_grpc

from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse

from threading import Lock
from concurrent import futures

from utilities.constants import MAX_CAMERAS

import time
import random


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


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_CAMERAS))
    object_detection_pb2_grpc.add_DetectorServicer_to_server(Server(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
