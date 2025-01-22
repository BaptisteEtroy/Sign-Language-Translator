import cv2
import mediapipe as mp
import numpy as np
import pickle  # For loading the trained model

def load_model(model_path):
    """Load the trained model."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def extract_landmarks(hand_landmarks, frame_shape):
    """Extract normalized hand landmarks."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])  # Normalized coordinates
    return np.array(landmarks).flatten()  # Flatten the array for the model

def main():
    # Load the pre-trained model
    model_path = "asl_hand_model.pkl"  # Replace with your trained model file
    model = load_model(model_path)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Flip the frame for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (required for MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks and recognize gestures if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark features
                landmarks = extract_landmarks(hand_landmarks, frame.shape)

                # Predict the letter using the trained model
                prediction = model.predict([landmarks])[0]

                # Draw bounding box around the hand
                x_min = int(min(lm.x for lm in hand_landmarks.landmark) * frame.shape[1])
                y_min = int(min(lm.y for lm in hand_landmarks.landmark) * frame.shape[0])
                x_max = int(max(lm.x for lm in hand_landmarks.landmark) * frame.shape[1])
                y_max = int(max(lm.y for lm in hand_landmarks.landmark) * frame.shape[0])

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Display the recognized letter
                cv2.putText(frame, f"Letter: {prediction}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Sign Translator', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()