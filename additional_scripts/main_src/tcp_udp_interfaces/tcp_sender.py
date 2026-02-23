import multiprocessing as mp
import socket
import cv2 as cv
import numpy as np


class TCP_server:
    def __init__(self, q_get:mp.Queue, ip_adress="localhost", port=6000, quality=100, format_img=".jpg", recv_bytes=1024):
        self.ip_adress, self.port = ip_adress, port
        self.q_get_queue, self.quality, self.format_img, self.recv_bytes = q_get, quality, format_img, recv_bytes



    def run(self):
        proc_mp = mp.Process(target=self.worker, args=(), daemon=True)
        proc_mp.start()

    def worker(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)

        try:
            sock.connect((self.ip_adress, self.port))

            while True:
                if not self.q_get_queue.empty():
                    frame = self.q_get_queue.get()
                    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), self.quality]
                    result, imgdecode = cv.imencode(self.format_img, frame, encode_param)

                    data = np.array(imgdecode)
                    sock.sendall(data.tobytes())
                    dt = sock.recv(self.recv_bytes)
                    if not dt:
                        break
                else:
                    print(f"Queue_empty_image")

            sock.close()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            sock.close()

