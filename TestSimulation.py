"""
Drives the car in the Udacity simulator's Autonomous Mode using a trained model.

Run this inside the same conda environment used for training/data collection
(cvi620), then in the simulator choose "Autonomous Mode".

Usage:
    python TestSimulation.py model/model.h5
"""
import argparse
import base64
from io import BytesIO

import eventlet
import socketio
import numpy as np
from PIL import Image
from flask import Flask
from tensorflow.keras.models import load_model

from preprocessing import img_preprocess

sio = socketio.Server()
app = Flask(__name__)
model = None
MAX_SPEED = 20


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        "steering_angle": str(steering_angle),
        "throttle": str(throttle),
    })


@sio.on("connect")
def connect(sid, environ):
    print("Connected:", sid)
    send_control(0, 0)


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        return

    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)

    image = img_preprocess(image)
    image = np.array([image])

    steering_angle = float(model.predict(image, verbose=0)[0][0])
    throttle = 1.0 - speed / MAX_SPEED

    print(f"steering={steering_angle:.4f} throttle={throttle:.4f} speed={speed:.2f}")
    send_control(steering_angle, throttle)


def main():
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained .h5 model")
    parser.add_argument("--port", type=int, default=4567)
    args = parser.parse_args()

    model = load_model(args.model)

    app_wrapped = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", args.port)), app_wrapped)


if __name__ == "__main__":
    main()
