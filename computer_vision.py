import cv2
import numpy as np
import pygame

def calibrate(eye_positions, screen_positions):
    """Compute the transformation from eye coordinates to screen coordinates."""
    eye_positions = np.array(eye_positions, dtype=np.float32)
    screen_positions = np.array(screen_positions, dtype=np.float32)

    # Find the affine transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(eye_positions, screen_positions)
    return transformation_matrix

def apply_transformation(transformation_matrix, eye_position):
    """Apply the transformation matrix to map eye position to screen position."""
    eye_position = np.array([eye_position], dtype=np.float32)
    transformed_position = cv2.transform(np.array([eye_position]), transformation_matrix)
    return int(transformed_position[0][0][0]), int(transformed_position[0][0][1])

def main():
    # Initialize Pygame
    pygame.init()

    # Full screen resolution
    screen_width, screen_height = 1512, 832
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Eye Tracker")
    clock = pygame.time.Clock()

    # Load the pre-trained Haar cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Camera setup
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Calibration data
    eye_positions = []
    screen_positions = [
        (screen_width // 2, screen_height // 2),  # Center
        (100, 100),  # Top-left
        (screen_width - 100, 100),  # Top-right
        (100, screen_height - 100),  # Bottom-left
        (screen_width - 100, screen_height - 100)  # Bottom-right
    ]

    transformation_matrix = None

    # Calibration step
    print("Calibration starting. Look at the points displayed on the screen.")
    for i, (sx, sy) in enumerate(screen_positions):
        calibration_done = False
        print(f"Look at point {i + 1}: ({sx}, {sy}). Press SPACE when ready.")
        while not calibration_done:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            # Display calibration point
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 0, 0), (sx, sy), 10)  # Draw calibration point
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    cap.release()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # Record eye position on space key press
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_region_gray = gray[y:y + h, x:x + w]
                        eyes = eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

                        if len(eyes) > 0:
                            # Record the center of the largest eye detected
                            ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[0]
                            cx, cy = ex + ew // 2, ey + eh // 2
                            eye_positions.append((cx, cy))
                            print(f"Recorded eye position: ({cx}, {cy})")
                            calibration_done = True

    # Calculate the transformation matrix
    transformation_matrix = calibrate(eye_positions, screen_positions)
    print("Calibration completed.")

    # Main tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_region_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            if len(eyes) > 0:
                # Use the largest eye detected
                ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[0]
                cx, cy = ex + ew // 2, ey + eh // 2

                # Map eye position to screen coordinates
                if transformation_matrix is not None:
                    gaze_x, gaze_y = apply_transformation(transformation_matrix, (cx, cy))

                    # Ensure gaze position is within screen bounds
                    gaze_x = min(max(gaze_x, 0), screen_width)
                    gaze_y = min(max(gaze_y, 0), screen_height)

                    # Draw gaze position on screen
                    screen.fill((0, 0, 0))
                    pygame.draw.circle(screen, (0, 255, 0), (gaze_x, gaze_y), 10)  # Green circle
                    pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                return

        clock.tick(60)  # Limit to 60 frames per second

if __name__ == "__main__":
    main()