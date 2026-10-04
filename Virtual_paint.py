import mediapipe as mp
import cv2
import numpy as np
import time

# Constants
ml = 150  # Left margin for tool panel
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Tool selector function
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

# Check if index finger is raised
def index_raised(yi, y9):
    return (y9 - yi) > 40

# Initialize MediaPipe
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Load and resize tools image
tools = cv2.imread("tools.png")
if tools is None:
    print("Error: 'tools.png' not found.")
    exit()
tools = cv2.resize(tools, (max_x - ml, max_y))
tools = tools.astype('uint8')

# Create white canvas
mask = np.ones((480, 640), dtype='uint8') * 255

# Start webcam
cap = cv2.VideoCapture(0)

# Create and optionally set full screen window
cv2.namedWindow("paint app", cv2.WINDOW_NORMAL)
# Uncomment the next line only if you're sure your system supports fullscreen
# cv2.setWindowProperty("paint app", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frm = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frm = cv2.flip(frm, 1)

    # Check mask size compatibility
    if frm.shape[:2] != mask.shape[:2]:
        mask = np.ones((frm.shape[0], frm.shape[1]), dtype='uint8') * 255

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            # Tool selection area
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()
                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("Your current tool set to:", curr_tool)
                    time_init = True
                    rad = 40
            else:
                time_init = True
                rad = 40

            xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
            y9 = int(i.landmark[9].y * 480)

            if curr_tool == "draw":
                if index_raised(yi, y9):
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                    prevx, prevy = x, y
                else:
                    prevx, prevy = x, y

            elif curr_tool == "line":
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)
                else:
                    if var_inits:
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)
                else:
                    if var_inits:
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "circle":
                if index_raised(yi, y9):
                    if not var_inits:
                        xii, yii = x, y
                        var_inits = True
                    radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                    cv2.circle(frm, (xii, yii), radius, (255, 255, 0), thick)
                else:
                    if var_inits:
                        radius = int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5)
                        cv2.circle(mask, (xii, yii), radius, (0, 255, 0), thick)
                        var_inits = False

            elif curr_tool == "erase":
                if index_raised(yi, y9):
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                    cv2.circle(mask, (x, y), 30, 255, -1)

    # Combine frame and mask
    op = cv2.bitwise_and(frm, frm, mask=mask)
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    # Draw tool panel
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    # Show current tool
    cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display output
    cv2.imshow("paint app", frm)

    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
