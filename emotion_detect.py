import cv2
from deepface import DeepFace

# Define emotion to color mapping (interpolated spectrum)
emotion_colors = {
    "angry": (0, 0, 255),       # Red
    "disgust": (0, 128, 255),   # Orange
    "fear": (128, 0, 128),      # Purple
    "sad": (0, 0, 200),         # Dark Blue
    "neutral": (255, 255, 255), # White
    "happy": (0, 255, 0),       # Green
    "surprise": (0, 255, 255)   # Cyan
}

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotion
    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Get dominant emotion and corresponding color
    emotion = results[0]['dominant_emotion']
    color = emotion_colors.get(emotion, (255, 255, 255))  # default white

    # Prepare text
    text = f'Emotion: {emotion.capitalize()}'
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    thickness = 2

    # Calculate position to center the text at bottom
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 30  # 30 pixels from bottom

    # Put text on frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Show the frame
    cv2.imshow("Emotion Recognition", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
