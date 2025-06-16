import cv2
import mediapipe as mp
import pyautogui
import random
import time
import util
from pynput.mouse import Button, Controller

mouse = Controller()
screen_width, screen_height = pyautogui.size()

hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

is_paused = False
prev_y = None
prev_x, prev_y_cursor = 0, 0
last_click_time = 0
click_delay = 1  # seconds

def get_index_tip(processed):
    if processed.multi_hand_landmarks:
        return processed.multi_hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_cursor(pos):
    global prev_x, prev_y_cursor
    if pos:
        x = int(pos.x * screen_width * 1.1)
        y = int(pos.y * screen_height * 1.15)
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        x = int((x + prev_x) / 2)
        y = int((y + prev_y_cursor) / 2)
        prev_x, prev_y_cursor = x, y
        pyautogui.moveTo(x, y)

def all_fingers_up(lm):
    return (
        lm[4][1] < lm[3][1] and
        lm[8][1] < lm[6][1] and
        lm[12][1] < lm[10][1] and
        lm[16][1] < lm[14][1] and
        lm[20][1] < lm[18][1]
    )

def is_left_click(lm, d): return util.get_angle(lm[5], lm[6], lm[8]) < 40 and util.get_angle(lm[9], lm[10], lm[12]) > 100 and d > 100
def is_right_click(lm, d): return util.get_angle(lm[9], lm[10], lm[12]) < 40 and util.get_angle(lm[5], lm[6], lm[8]) > 100 and d > 100
def is_double_click(lm, d): return util.get_angle(lm[5], lm[6], lm[8]) < 40 and util.get_angle(lm[9], lm[10], lm[12]) < 40 and d > 100
def is_screenshot(lm, d): return util.get_angle(lm[5], lm[6], lm[8]) < 40 and util.get_angle(lm[9], lm[10], lm[12]) < 40 and d < 50

def detect(frame, points, processed):
    global is_paused, prev_y, last_click_time

    if len(points) < 21:
        return

    if all_fingers_up(points):
        is_paused = True
        cv2.putText(frame, "PAUSED", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 255), 2)
        return
    elif is_paused:
        is_paused = False

    if is_paused:
        return

    now = time.time()
    idx_tip = get_index_tip(processed)
    thumb_idx_dist = util.get_distance([points[4], points[5]])

    idx_up = points[8][1] < points[6][1]
    mid_up = points[12][1] < points[10][1]

    if is_left_click(points, thumb_idx_dist) and now - last_click_time > click_delay:
        mouse.click(Button.left)
        last_click_time = now
        cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif is_right_click(points, thumb_idx_dist) and now - last_click_time > click_delay:
        mouse.click(Button.right)
        last_click_time = now
        cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif is_double_click(points, thumb_idx_dist) and now - last_click_time > click_delay:
        pyautogui.doubleClick()
        last_click_time = now
        cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    elif is_screenshot(points, thumb_idx_dist) and now - last_click_time > click_delay:
        shot = pyautogui.screenshot()
        shot.save(f'screenshot_{random.randint(1000,9999)}.png')
        last_click_time = now
        cv2.putText(frame, "Screenshot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    elif idx_up and mid_up:
        if idx_tip:
            curr_y = idx_tip.y
            if prev_y is not None:
                delta = curr_y - prev_y
                if delta < -0.02:
                    pyautogui.scroll(30)
                    cv2.putText(frame, "Scroll Up", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif delta > 0.02:
                    pyautogui.scroll(-30)
                    cv2.putText(frame, "Scroll Down", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            prev_y = curr_y

    elif idx_up and not mid_up:
        if util.get_angle(points[5], points[6], points[8]) > 80:
            move_cursor(idx_tip)

def main():
    draw = mp.solutions.drawing_utils
    cam = cv2.VideoCapture(0)

    try:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            points = []
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                points = [(l.x, l.y) for l in lm.landmark]

            detect(frame, points, result)
            cv2.imshow("Virtual Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
