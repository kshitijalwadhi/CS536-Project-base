import grpc

import object_detection_pb2_grpc
from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse


class Client:
    def __init__(self, server_address, client_fps):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = object_detection_pb2_grpc.DetectorStub(self.channel)

        req = InitRequest()
        resp: InitResponse = self.stub.init_client(req)
        self.client_id = resp.client_id

        print("Client ID: {}".format(self.client_id))

        self.client_fps = client_fps

    def close_connection(self):
        req = CloseRequest(client_id=self.client_id)
        resp: CloseResponse = self.stub.close_connection(req)
        print("Close connection for client: {}".format(resp.client_id))


if __name__ == '__main__':
    client = Client('localhost:50051', 30)
    client.close_connection()
