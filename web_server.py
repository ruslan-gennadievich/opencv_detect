import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np

class ImageUploadServer:    
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/upload':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)                
                try:
                    np_arr = np.frombuffer(post_data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_ANYCOLOR)
                    
                    if image is None:
                        raise ValueError("Failed to decode image")
                    
                    self.server.uploaded_CV_Image = image                    
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"message": "Image received successfully"}')
                except Exception as e:
                    print (e)
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = '{"error": "' + str(e) + '"}'
                    self.wfile.write(response.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
    
    def __init__(self, host='0.0.0.0', port=5001):
        self.server_address = (host, port)
        self.httpd = HTTPServer(self.server_address, self.RequestHandler)
        self.httpd.uploaded_CV_Image = None
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
    
    def start(self):
        print(f"Starting server on {self.server_address[0]}:{self.server_address[1]}")
        self.server_thread.start()
    
    def run_server(self):
        self.httpd.serve_forever()
    
    def stop(self):
        self.httpd.shutdown()
        self.server_thread.join()
        print("Server stopped.")

    @property
    def uploaded_CV_Image(self):
        return self.httpd.uploaded_CV_Image

    @uploaded_CV_Image.setter
    def uploaded_CV_Image(self, value):
        self.httpd.uploaded_CV_Image = value
