import cv2
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("Kolam_v4.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Flag to control capturing
capturing = True


def generate_frames():
    global capturing
    while capturing:
        success, frame = cap.read()
        if not success:
            print("Error: Couldn't read from the webcam.")
            break

        # YOLO prediction
        results = model.predict(source=frame, conf=0.5)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates and class info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                label = f'Class: {model.names[int(cls)]}, Conf: {conf:.2f}'

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release resources when capturing stops
    cap.release()
    cv2.destroyAllWindows()


@app.get('/livevideo')
async def live_video_stream():
    """
    Route to stream live video with object detection.
    """
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
