import grpc

import object_detection_pb2_grpc
from object_detection_pb2 import InitRequest, InitResponse, CloseRequest, CloseResponse, DetectRequest, DetectResponse

from utilities.helpers import draw_result, yield_frames_from_video
from utilities.constants import IMG_SIZE

import time

from imutils.video import FileVideoStream
import cv2
import pickle


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

    def send_video(self):
        print("Sending video...")
        time.sleep(1)

        vs = FileVideoStream("data/sample.mp4").start()

        try:
            sequence_number = 0
            for img in yield_frames_from_video(vs):
                sequence_number += 1
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                jpg = cv2.imencode('.jpg', img)[1]
                frame_data = pickle.dumps(jpg)

                req = DetectRequest(
                    client_id=self.client_id,
                    fps=self.client_fps,
                    sequence_number=sequence_number,
                    frame_data=frame_data
                )

                resp: DetectResponse = self.stub.detect(req)

                req_dropped: bool = resp.req_dropped

                if req_dropped:
                    print("Request with sequence number was {} dropped".format(sequence_number))
                    cv2.imshow("Result", img)
                else:
                    result = pickle.loads(resp.bboxes.data)
                    if result:
                        display = draw_result(img, result)
                        cv2.imshow("Result", display)
                    else:
                        cv2.imshow("Result", img)

                cv2.waitKey(1)

        except grpc._channel._Rendezvous as err:
            print("Error: {}".format(err))
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            vs.stop()


if __name__ == '__main__':
    client = Client('localhost:50051', 30)
    client.send_video()
    client.close_connection()
