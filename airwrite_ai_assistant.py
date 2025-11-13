import argparse
import json
import math
import os
import re
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration constants
WINDOW_TITLE = "AirWrite AI Assistant"
CANVAS_SIZE = (480, 640)  # (height, width)
TRAIL_LENGTH = 512
DRAW_COLOR = (0, 255, 0)
FINGER_TIP_ID = 8  # Mediapipe landmark index for the index fingertip
THUMB_TIP_ID = 4
PINCH_ON_THRESHOLD = 0.09
PINCH_OFF_THRESHOLD = 0.12
PINCH_SMOOTHING = 0.25
POINT_SMOOTHING = 0.35
DEFAULT_MODEL_PATH = os.path.join("models", "airwrite_cnn.h5")
DEFAULT_LABEL_MAP_PATH = os.path.join("models", "label_map.json")
DEFAULT_CLASS_MAP = {
    idx: char
    for idx, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/()=")
}


def load_classifier(model_path: str, label_map_path: Optional[str]) -> Tuple[Optional[object], Dict[int, str]]:
    """Load the trained Keras model and optional label map for decoding predictions."""
    model = None
    class_map = DEFAULT_CLASS_MAP.copy()

    if os.path.isfile(model_path):
        model = load_model(model_path)
    else:
        print(f"[WARN] Model file not found at {model_path}. Prediction will be skipped.")

    if label_map_path and os.path.isfile(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as mapping_file:
                loaded_map = json.load(mapping_file)
            class_map = {int(k): v for k, v in loaded_map.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[WARN] Failed to load label map: {exc}. Falling back to default map.")

    return model, class_map


def preprocess_canvas(canvas: np.ndarray) -> np.ndarray:
    """Convert the drawing canvas to the format expected by the classifier."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    inverted = 1.0 - normalized  # Ensure white background and dark strokes
    image = img_to_array(inverted)
    return np.expand_dims(image, axis=0)


def predict_character(model, class_map: Dict[int, str], canvas: np.ndarray) -> Optional[str]:
    """Predict the character drawn on the canvas using the CNN."""
    if model is None:
        return None
    processed = preprocess_canvas(canvas)
    predictions = model.predict(processed, verbose=0)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    return class_map.get(predicted_index)


def evaluate_text_buffer(text_buffer: List[str]) -> Tuple[str, Optional[str]]:
    """Evaluate the buffer and produce a display message for math or plain text."""
    text = "".join(text_buffer).strip()
    if not text:
        return "", None

    math_pattern = re.compile(r"^[0-9+\-*/().=\s]+$")
    if math_pattern.fullmatch(text):
        try:
            safe_globals = {"__builtins__": {}}
            safe_locals = {}
            result = eval(text, safe_globals, safe_locals)
            return text, f"{text} = {result}"
        except Exception as exc:  # noqa: BLE001
            return text, f"Math error: {exc}"

    return text, text


def detect_index_finger_tip(hand_landmarks, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Convert Mediapipe index fingertip coordinates into pixel coordinates."""
    height, width = frame_shape
    landmark = hand_landmarks.landmark[FINGER_TIP_ID]
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def append_trail_point(trail: Deque[Optional[Tuple[int, int]]], point: Optional[Tuple[int, int]]) -> None:
    """Append the latest fingertip point (or separator) to the deque used for drawing."""
    if point is None:
        if trail and trail[0] is None:
            return
        trail.appendleft(None)
    else:
        trail.appendleft(point)
    if len(trail) > TRAIL_LENGTH:
        trail.pop()


def draw_trail(trail: List[Optional[Tuple[int, int]]], image: np.ndarray, color: Tuple[int, int, int], thickness: int = 6) -> None:
    """Draw the fingertip trail both on the live frame and the persistent canvas."""
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        cv2.line(image, trail[i - 1], trail[i], color, thickness)


def compute_pinch_distance(hand_landmarks) -> float:
    """Return the 3D distance between thumb and index fingertips (normalized units)."""
    thumb = hand_landmarks.landmark[THUMB_TIP_ID]
    index = hand_landmarks.landmark[FINGER_TIP_ID]
    dx = index.x - thumb.x
    dy = index.y - thumb.y
    dz = index.z - thumb.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def update_drawing_state(distance: float, current_state: bool) -> bool:
    """Flip the drawing state using hysteresis to reduce jitter while pinching."""
    if current_state:
        return distance < PINCH_OFF_THRESHOLD
    return distance < PINCH_ON_THRESHOLD


def render_ui(
    frame: np.ndarray,
    canvas: np.ndarray,
    text_buffer: List[str],
    result_text: Optional[str],
    pen_down: bool,
    use_pinch_control: bool,
    pinch_distance: Optional[float],
) -> np.ndarray:
    """Combine the live frame, drawing canvas, and overlay text for the UI."""
    overlay = frame.copy()
    buffer_text = "".join(text_buffer)
    cv2.putText(overlay, f"Buffer: {buffer_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    if result_text:
        cv2.putText(overlay, f"Result: {result_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    mode_label = "Pinch" if use_pinch_control else "Manual"
    state_label = "pen down" if pen_down else "pen up"
    pinch_label = f" | pinch: {pinch_distance:.3f}" if pinch_distance is not None else ""
    cv2.putText(
        overlay,
        f"Mode: {mode_label} ({state_label}){pinch_label}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 40, 40),
        2,
    )

    instructions = [
        "pinch (or d in manual mode) = pen down",
        "t = toggle pinch/manual",
        "s = save & predict",
        "space = add space",
        "enter = evaluate buffer",
        "b = backspace",
        "c = clear",
        "q = quit",
    ]

    base_y = frame.shape[0] - 120
    for idx, line in enumerate(instructions):
        y_pos = base_y + idx * 22
        cv2.putText(
            overlay,
            line,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
    stacked = np.hstack((overlay, canvas_resized))
    return stacked


def run_airwrite(model_path: str, label_map_path: Optional[str]) -> None:
    """Main application loop handling video capture, drawing, prediction, and UI."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Ensure a camera is connected and accessible.")

    canvas = np.ones((*CANVAS_SIZE, 3), dtype=np.uint8) * 255
    fingertip_trail: Deque[Optional[Tuple[int, int]]] = deque(maxlen=TRAIL_LENGTH)
    text_buffer: List[str] = []
    last_result: Optional[str] = None
    use_pinch_control = True
    pen_down = False
    canvas_last_point: Optional[Tuple[int, int]] = None
    pinch_distance_smoothed: Optional[float] = None
    smoothed_point: Optional[Tuple[float, float]] = None

    model, class_map = load_classifier(model_path, label_map_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            previous_pen_state = pen_down
            fingertip_point: Optional[Tuple[int, int]] = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                pinch_distance = compute_pinch_distance(hand_landmarks)
                if pinch_distance_smoothed is None:
                    pinch_distance_smoothed = pinch_distance
                else:
                    pinch_distance_smoothed = (
                        pinch_distance_smoothed * (1 - PINCH_SMOOTHING) + pinch_distance * PINCH_SMOOTHING
                    )

                fingertip_raw = detect_index_finger_tip(hand_landmarks, (frame.shape[0], frame.shape[1]))
                if fingertip_raw:
                    if smoothed_point is None:
                        smoothed_point = (float(fingertip_raw[0]), float(fingertip_raw[1]))
                    else:
                        smoothed_point = (
                            smoothed_point[0] * (1 - POINT_SMOOTHING) + fingertip_raw[0] * POINT_SMOOTHING,
                            smoothed_point[1] * (1 - POINT_SMOOTHING) + fingertip_raw[1] * POINT_SMOOTHING,
                        )
                    fingertip_point = (int(smoothed_point[0]), int(smoothed_point[1]))
                else:
                    smoothed_point = None

                if use_pinch_control and pinch_distance_smoothed is not None:
                    pen_down = update_drawing_state(pinch_distance_smoothed, pen_down)

                indicator_color = (0, 0, 255) if pen_down else (0, 255, 255)
                if fingertip_point:
                    cv2.circle(frame, fingertip_point, 8, indicator_color, -1)

                for handLms in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            else:
                smoothed_point = None
                pinch_distance_smoothed = None if use_pinch_control else pinch_distance_smoothed
                if previous_pen_state:
                    append_trail_point(fingertip_trail, None)
                if use_pinch_control:
                    pen_down = False

            if pen_down and fingertip_point:
                append_trail_point(fingertip_trail, fingertip_point)
                if canvas_last_point is not None:
                    cv2.line(canvas, canvas_last_point, fingertip_point, (0, 0, 0), 6)
                canvas_last_point = fingertip_point
            else:
                if previous_pen_state and (not pen_down or fingertip_point is None):
                    append_trail_point(fingertip_trail, None)
                if not pen_down or fingertip_point is None:
                    canvas_last_point = None

            trail_points = list(fingertip_trail)
            draw_trail(trail_points, frame, DRAW_COLOR, thickness=4)

            ui_frame = render_ui(
                frame,
                canvas,
                text_buffer,
                last_result,
                pen_down,
                use_pinch_control,
                pinch_distance_smoothed,
            )
            cv2.imshow(WINDOW_TITLE, ui_frame)

            key = cv2.waitKey(1) & 0xFF

            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord("q"):
                break
            if key == ord("c"):
                canvas.fill(255)
                fingertip_trail.clear()
                canvas_last_point = None
                smoothed_point = None
                if use_pinch_control:
                    pen_down = False
            if key == ord("s"):
                filename = "char.png"
                cv2.imwrite(filename, canvas)
                print(f"[INFO] Saved canvas to {filename}")
                predicted = predict_character(model, class_map, canvas)
                if predicted:
                    text_buffer.append(predicted)
                    print(f"[INFO] Recognized character: {predicted}")
                    last_result = None
                else:
                    print("[INFO] Unable to predict character. Ensure model is loaded.")
                canvas.fill(255)
                fingertip_trail.clear()
                canvas_last_point = None
                smoothed_point = None
                if use_pinch_control:
                    pen_down = False
            if key == 32:  # Spacebar
                text_buffer.append(" ")
                last_result = None
            if key in (13, 10):  # Enter key variants
                _, last_result = evaluate_text_buffer(text_buffer)
            if key == ord("b"):
                # Optional backspace support
                if text_buffer:
                    text_buffer.pop()
                    last_result = None
            if key == ord("t"):
                use_pinch_control = not use_pinch_control
                append_trail_point(fingertip_trail, None)
                canvas_last_point = None
                smoothed_point = None
                pinch_distance_smoothed = None
                pen_down = False
                mode_name = "pinch" if use_pinch_control else "manual"
                print(f"[INFO] Switched to {mode_name} drawing mode.")
            if key == ord("d"):
                if not use_pinch_control:
                    pen_down = not pen_down
                    state = "down" if pen_down else "up"
                    if not pen_down:
                        append_trail_point(fingertip_trail, None)
                        canvas_last_point = None
                    print(f"[INFO] Manual pen is now {state}.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for model configuration."""
    parser = argparse.ArgumentParser(description="Run the AirWrite AI Assistant application.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained Keras model (.h5).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=DEFAULT_LABEL_MAP_PATH,
        help="Optional path to a label map JSON file mapping class indices to characters.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_airwrite(args.model, args.labels)
