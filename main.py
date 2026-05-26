from pathlib import Path
from urllib.request import urlretrieve
import math
import random

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


class ChaosParticle:
    def __init__(self, x, y, size_base):
        self.x = x
        self.y = y
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-10, 2)

        self.life = 1.0
        self.decay = random.uniform(0.04, 0.08)
        self.size = random.uniform(size_base * 0.5, size_base * 1.5)

        self.wobble_speed = random.uniform(0.1, 0.4)
        self.wobble_offset = random.uniform(0, math.pi * 2)

    def update(self):
        self.x += self.vx + math.sin(self.life * 15.0 * self.wobble_speed + self.wobble_offset) * 6
        self.y += self.vy
        self.life -= self.decay
        return self.life > 0


def get_pt(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def ensure_model_exists():
    if not MODEL_PATH.exists():
        print("Downloading hand landmarker model...")
        urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def create_landmarker():
    model_path = ensure_model_exists()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    hands = create_landmarker()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("╔═══════════════════════════════════════════════════╗")
    print("║   Realistic Scarlet Witch Chaos Magic VFX 🔴✨    ║")
    print("╚═══════════════════════════════════════════════════╝")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    h, w, c = frame.shape
    plasma_buffer = np.zeros((h, w, 3), dtype=np.float32)
    particles = []
    timestamp_ms = 0
    rise_matrix = np.float32([[1, 0, 0], [0, 1, -4]])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = hands.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                wrist = get_pt(hand_landmarks[0], w, h)
                mcp9 = get_pt(hand_landmarks[9], w, h)
                hand_size = max(
                    10,
                    math.sqrt((wrist[0] - mcp9[0]) ** 2 + (wrist[1] - mcp9[1]) ** 2),
                )

                tips = [4, 8, 12, 16, 20]
                palm = get_pt(hand_landmarks[9], w, h)

                for tip_idx in tips:
                    pt = get_pt(hand_landmarks[tip_idx], w, h)
                    for _ in range(random.randint(1, 2)):
                        particles.append(ChaosParticle(pt[0], pt[1], hand_size * 0.15))

                for _ in range(5):
                    angle = random.uniform(0, math.pi * 2)
                    radius = random.uniform(0, hand_size * 0.5)
                    ox = math.cos(angle) * radius
                    oy = math.sin(angle) * radius
                    particles.append(
                        ChaosParticle(palm[0] + ox, palm[1] + oy, hand_size * 0.35)
                    )

        plasma_buffer = cv2.warpAffine(plasma_buffer, rise_matrix, (w, h))

        temp_layer = np.zeros_like(plasma_buffer)
        alive_particles = []
        for particle in particles:
            if particle.update():
                alive_particles.append(particle)
                alpha = particle.life
                x, y = int(particle.x), int(particle.y)

                cv2.circle(temp_layer, (x, y), int(particle.size * 1.8), (10 * alpha, 10 * alpha, 80 * alpha), -1)
                cv2.circle(temp_layer, (x, y), int(particle.size * 1.0), (30 * alpha, 30 * alpha, 220 * alpha), -1)
                cv2.circle(temp_layer, (x, y), int(particle.size * 0.35), (150 * alpha, 150 * alpha, 255 * alpha), -1)

        particles = alive_particles
        plasma_buffer += temp_layer
        plasma_buffer = cv2.GaussianBlur(plasma_buffer, (7, 7), 0)
        plasma_buffer *= 0.82

        plasma_visual = np.clip(plasma_buffer, 0, 255).astype(np.uint8)
        output_frame = cv2.add(frame, plasma_visual)

        intensity = min(1.0, len(particles) / 400.0)
        if intensity > 0.05:
            ambient = np.zeros_like(frame)
            ambient[:, :, 2] = int(45 * intensity)
            ambient[:, :, 0] = int(10 * intensity)
            output_frame = cv2.add(output_frame, ambient)

        cv2.imshow("Scarlet Witch Chaos Magic VFX", output_frame)

        if len(particles) > 600:
            particles = particles[-400:]

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
