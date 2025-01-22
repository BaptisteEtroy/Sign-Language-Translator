import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from models import load_asl_model  # Import the load function from models.py

# Load the model
device = torch.device("cpu")  # or "mps" if Apple Silicon GPU is available
model = load_asl_model("asl_model_pytorch.pth", device=device, num_classes=29)

# Create transforms matching your training pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Label list: must match alphabetical order from your dataset
asl_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # If a hand is detected, draw landmarks and classify
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box
            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * frame.shape[1])-200
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * frame.shape[0])-200
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * frame.shape[1])+200
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * frame.shape[0])+200

            # Extract ROI
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size != 0:
                # Convert ROI to tensor
                roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                tensor_img = transform(roi_rgb).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(tensor_img)
                _, pred_idx = torch.max(outputs, 1)
                letter = asl_labels[pred_idx.item()]

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f"Letter: {letter}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("ASL Translator (PyTorch)", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()