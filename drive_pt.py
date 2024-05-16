import socketio
import eventlet
from flask import Flask
import numpy as np
import base64
import cv2
import torch
from torch import load
from torchvision import transforms
from model import Model
MAX_SPEED = 30
MIN_SPEED = 28
speed_limit = MAX_SPEED
model = None
print("Starting server...")
sio = socketio.Server()
app = Flask(__name__)
print("Server started.")
def preprocess_image_with_cv2(encoded_image):
    image_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (200, 66))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0).type(torch.float32)

    return img_tensor

@sio.on("telemetry")
def telemetry(sid, data):
    global model, speed_limit
    if data is not None:
        speed = float(data["speed"])
        throttle = float(data["throttle"])
        image_tensor = preprocess_image_with_cv2(data["image"])  
        cv2.imshow("Front camera", image_tensor[0].permute(1, 2, 0).numpy())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

        steering_angle = 0.0
        if model is not None:
            steering_angle = float(model(image_tensor))
        
        if speed > speed_limit:
            speed_limit = MIN_SPEED
        else:
            speed_limit = MAX_SPEED

        throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

        print(f"Steering angle: {steering_angle:.5f}. Throttle: {throttle:10.6f}")
        send_control(steering_angle, throttle)
    else:
        send_control(0, 0)
@sio.on("connect")
def connect(sid, environ):
    print("I am connected to the Simulator!", sid)
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data = {
        "steering_angle": steering_angle.__str__(), 
        "throttle": throttle.__str__()
    })
if __name__ == "__main__":
    print("Loading model...")
    model = Model()
    model_path = "best_model.pth"
    model.load_state_dict(load(model_path))
    model.eval()
    print("Loaded model.")
    print("Middleware")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
