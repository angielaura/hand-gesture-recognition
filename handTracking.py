import cv2 # Used to handle webcam and displaying images
import mediapipe as mp # Used for detecting hands and drawing landmarks

mp_hands = mp.solutions.hands   # Access mediapipe hand tracking model
hands = mp_hands.Hands()    # Initialize the hand tracking model, it detect and tracks hands in video frames
mp_draw = mp.solutions.drawing_utils # Provides function to draw the hand landmarks on the images

cap = cv2.VideoCapture(0) # Opens default web cam (index 0), will continuously captures frames from webcam

# Define the function to recognize hand gestures with hand_landmarks (hand key points detected by Mediapipe) as parameters
def recognize_gestures(hand_landmarks) :
    fingers = []    # Empty array to store wether each fingers is open (1) or closed (0)
    tip_ids = [4, 8, 12, 16, 20]    # This is the id of each fingers, thumb tip is at point 4, index tip is at point 8, and so on

    # BELOW ARE THE BASE CASE
    # Detect if the thumb is open or not
    # We use x axis do detect the thumb since it moves sideways
    # Below line is basically said "if thumb tip is to the left of the thumb base"
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x :
        fingers.append(1)   # Then it is opened
    else : 
        fingers.append(0)   # Else it is closed

    # Now for the other 4 fingers
    # Loop through the index to pinky (1 to 4) and checks if it is open or not, we use y axis
    # Below line is basically saying "If the finger tip is above its middle joint"
    for i in range(1, 5) :
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y :
            fingers.append(1)   # "Then it is opened"
        else : 
            fingers.append(0)   # "Else it is closed"

    # Check the distance between thumb and index
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 0, 1, 0, 0] :
        return "F- You"
    elif fingers == [1, 1, 0, 0, 0] :
        return "Saranghae"
    elif fingers == [0, 0, 1, 1, 1] :
        return "Okay!"
    else:
        return "Unknown Gesture"

# This loop runs as long as the camera is opened, ensuring cam is opened while program is running
while cap.isOpened():
    ret, frame = cap.read()     # Read every single frame in the web cam
    if not ret: # Returns true if the frame is successfully read
        break   # If false, then no frame is captured, loop breaks to prevent error

    # OpenCV uses BGR format (Blue Green Red), but Mediapipe needs RGB (Red Green Blue)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # So this converts the frame from BGR to RGB
    results = hands.process(rgb_frame) # Detects hands in frame, then results will contain the detected hand landmark

    # results.multi_hand_landmarks contains a list of detected hands
    # This loop will loop through each detected hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_draw.draw_landmarks() will draw 21 key points on the detected hand
            # mp_hands.HAND_CONNECTIONS will connect the points to form a hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gestures(hand_landmarks)    # Pass the points to the function

            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the video feed with detected hand landmarks
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on 'q' key press, and will wait for keypress for every 1ms
    # If user press 'q', then loop breaks and the program will quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()   # Stops the webcam and releases it to other program
cv2.destroyAllWindows() # Closes all OpenCV windows
