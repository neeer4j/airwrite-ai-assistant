## AirWrite Assistant

AirWrite Assistant lets you sketch characters or quick math expressions in mid-air using nothing more than your hand. Mediapipe tracks your index fingertip, OpenCV renders both the live trail and a clean canvas, and a TensorFlow/Keras classifier attempts to recognize the symbol you just drew. When the buffer forms a math expression (e.g. `2+3*4`) the app evaluates it for you using standard operator precedence.

### Prerequisites

Install the required packages (preferably inside a virtual environment):

```bash
pip install opencv-python mediapipe numpy tensorflow
```

### Run

```bash
python airwrite_ai_assistant.py
```

### Controls

- `pinch` index finger and thumb together to draw; release to reposition (manual toggle available in-app).
- `s` &mdash; save the current canvas to `char.png`, run recognition, and append the prediction.
- `space` &mdash; insert a space in the text buffer.
- `enter` &mdash; evaluate the buffer; math expressions show their result.
- `c` &mdash; clear the canvas and fingertip trail.
- `b` &mdash; backspace the latest character.
- `t` &mdash; toggle between pinch-to-draw and manual pen control modes.
- `q` &mdash; quit the application.

### Tech Stack

- **OpenCV** for video capture, drawing overlays, and compositing the UI.
- **MediaPipe Hands** for real-time hand landmark tracking.
- **TensorFlow / Keras** for the convolutional neural network classifier.
- **NumPy** for image preprocessing before inference.
