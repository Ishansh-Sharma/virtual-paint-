import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Color variables
selected_color = None  # No default color, drawing will happen only after a color is selected
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
drawing = False

# Define color selection rectangles and "Clear All" button
color_tabs = {
    "red": (10, 10, 110, 60),  # (x1, y1, x2, y2)
    "green": (130, 10, 230, 60),
    "blue": (250, 10, 350, 60)
}
clear_tab = (370, 10, 470, 60)  # Clear All button coordinates


# Create a blank canvas to draw on
def create_blank_canvas(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)  # Create a black canvas


def is_fist(landmarks, h, w):
    """
    Detect if the hand is in a fist by checking if all the fingers are folded.
    Returns True if a fist is detected.
    """
    # Thumb tip (4), Index tip (8), Middle tip (12), Ring tip (16), Pinky tip (20)
    finger_tips = [landmarks.landmark[4], landmarks.landmark[8], landmarks.landmark[12], landmarks.landmark[16],
                   landmarks.landmark[20]]
    # Base of the fingers: Index (6), Middle (10), Ring (14), Pinky (18)
    finger_bases = [landmarks.landmark[6], landmarks.landmark[10], landmarks.landmark[14], landmarks.landmark[18]]

    for tip, base in zip(finger_tips[1:], finger_bases):  # Skip thumb for simplicity
        if tip.y < base.y:  # If any fingertip is higher than its base, hand is not in a fist
            return False
    return True


cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None  # To track previous finger position
canvas = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Initialize canvas if it doesn't exist
    if canvas is None:
        canvas = create_blank_canvas(h, w)  # Create a blank black canvas

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # Draw color selection rectangles and "Clear All" button
    cv2.rectangle(frame, (color_tabs["red"][0], color_tabs["red"][1]), (color_tabs["red"][2], color_tabs["red"][3]),
                  red_color, -1)
    cv2.rectangle(frame, (color_tabs["green"][0], color_tabs["green"][1]),
                  (color_tabs["green"][2], color_tabs["green"][3]), green_color, -1)
    cv2.rectangle(frame, (color_tabs["blue"][0], color_tabs["blue"][1]), (color_tabs["blue"][2], color_tabs["blue"][3]),
                  blue_color, -1)
    cv2.rectangle(frame, (clear_tab[0], clear_tab[1]), (clear_tab[2], clear_tab[3]), (255, 255, 255), -1)
    cv2.putText(frame, "Clear", (clear_tab[0] + 10, clear_tab[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_tip_x = int(index_finger_tip.x * w)  # Convert normalized coordinates to pixel coordinates
            index_finger_tip_y = int(index_finger_tip.y * h)

            # Detect if the hand is in a fist, clear the color if true
            if is_fist(hand_landmarks, h, w):
                selected_color = None
                drawing = False
                prev_x, prev_y = None, None  # Reset previous points
                continue  # Skip drawing if fist is detected

            # Check if the finger is hovering over the color tabs to change color
            if color_tabs["red"][0] < index_finger_tip_x < color_tabs["red"][2] and color_tabs["red"][
                1] < index_finger_tip_y < color_tabs["red"][3]:
                selected_color = red_color
            elif color_tabs["green"][0] < index_finger_tip_x < color_tabs["green"][2] and color_tabs["green"][
                1] < index_finger_tip_y < color_tabs["green"][3]:
                selected_color = green_color
            elif color_tabs["blue"][0] < index_finger_tip_x < color_tabs["blue"][2] and color_tabs["blue"][
                1] < index_finger_tip_y < color_tabs["blue"][3]:
                selected_color = blue_color

            # Check if the finger is hovering over the "Clear All" button to clear the canvas
            if clear_tab[0] < index_finger_tip_x < clear_tab[2] and clear_tab[1] < index_finger_tip_y < clear_tab[3]:
                canvas = create_blank_canvas(h, w)  # Clear the canvas by resetting to a blank black canvas

            # If a color is selected and drawing is enabled, draw continuously
            if selected_color:
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = index_finger_tip_x, index_finger_tip_y  # Set initial position
                else:
                    # Draw lines between the previous and current fingertip positions for continuous drawing
                    cv2.line(canvas, (prev_x, prev_y), (index_finger_tip_x, index_finger_tip_y), selected_color, 10)
                    prev_x, prev_y = index_finger_tip_x, index_finger_tip_y  # Update position

    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the resulting frame
    cv2.imshow("Hand Drawing", frame)

    # Break the loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
