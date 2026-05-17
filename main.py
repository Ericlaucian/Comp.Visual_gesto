import mediapipe as mp
import cv2
import time
import math
import pyautogui as pag
import numpy as np

MAX_NUM_HANDS = 2

class Dot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# ============== Configurações da API do MediaPipe ================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
global last_distance 
last_distance = 0

def callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# ================== Configurações do modelo de detecção de mãos =================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=callback,
    num_hands=2
)


#================================= Funções auxiliares =============================================


def put_text(red, green, blue, texto, cx, cy):
    cv2.putText(
        img=frame, 
        text= texto,
        org=(cx, cy),          # Posição (x, y) em pixels
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1,           # Tamanho da fonte
        color=(blue, green, red), # Cor em BGR (Branco neste caso)
        thickness=2            # Espessura da linha das letras
    )

def circle_design(x, y):
    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

def hip(dx, dy):
    return math.sqrt(math.fabs(math.pow(dx, 2) + math.pow(dy, 2)))

def polegar(hand_landmarks, palma):
    polegar_palma = hip(hand_landmarks[4].x - palma.x/w, hand_landmarks[4].y - palma.y/h)
    polegar_base = hip(hand_landmarks[4].x - hand_landmarks[2].x, hand_landmarks[4].y - hand_landmarks[2].y)
    polegar_ind_base = hip(hand_landmarks[4].x - hand_landmarks[5].x, hand_landmarks[4].y - hand_landmarks[5].y)

    return polegar_palma > polegar_base and polegar_palma < polegar_ind_base * 2

def finger(hand_landmarks, palma, i):
    dedo_palma = hip(hand_landmarks[i].x - palma.x/w, hand_landmarks[i].y - palma.y/h)
    dedo_base = hip(hand_landmarks[i].x - hand_landmarks[i-2].x, hand_landmarks[i].y - hand_landmarks[i-2].y)
    dedo_ind_base = hip(hand_landmarks[i].x - hand_landmarks[5].x, hand_landmarks[i].y - hand_landmarks[5].y)   

    return dedo_palma > dedo_base and dedo_palma > dedo_ind_base

def dedos_left(palma):
    num_dedos = 0

    if latest_result:
        for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            label = latest_result.handedness[idx][0].category_name

            if label == "Left":
                if polegar(hand_landmarks, palma):
                    num_dedos += 1

            for i in [8, 12, 16, 20]:
                if label == "Left" and finger(hand_landmarks, palma, i):
                    num_dedos += 1
                
    return num_dedos

def dedos_right(palma):
    num_dedos = 0

    if latest_result:
        for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            label = latest_result.handedness[idx][0].category_name

            if label == "Right":
                if polegar(hand_landmarks, palma):
                    num_dedos += 1

            for i in [8, 12, 16, 20]:
                if label == "Right" and finger(hand_landmarks, palma, i):
                    num_dedos += 1
                
    return num_dedos

def mouse_control(palma):
    window_width, window_height = pag.size()
    mouse_x = np.interp(hand_landmarks[8].x, [0, w], [0, window_width])
    mouse_y = np.interp(hand_landmarks[8].y, [0, h], [0, window_height])
    if finger(hand_landmarks, palma, 8) and not finger(hand_landmarks, palma, 12) and not finger(hand_landmarks, palma, 16) and not finger(hand_landmarks, palma, 20):
        pag.moveTo(mouse_x * 900, mouse_y *900)
        if  polegar( hand_landmarks, palma):
            pag.click()
            time.sleep(0.7)
            if polegar( hand_landmarks, palma):
                pag.doubleClick()
                time.sleep(1)
                if polegar( hand_landmarks, palma):
                    pag.doubleClick()
                    pag.click()
                    time.sleep(2)
                

def scroll_control(palma):
    
    if finger(hand_landmarks, palma, 8) and finger(hand_landmarks, palma, 12) and not finger(hand_landmarks, palma, 16) and not finger(hand_landmarks, palma, 20):
        if polegar(hand_landmarks, palma):
            pag.scroll(50)
        else:
            pag.scroll(-50)


def side_arrow(palma):
    dx = hand_landmarks[12].x - hand_landmarks[0].x
    dy = hand_landmarks[12].y - hand_landmarks[0].y
    if np.arctan2(dy, dx) < -1.8:
        pag.press('left')
        time.sleep(0.8)
    elif np.arctan2(dy, dx) > -1.2:
        pag.press('right')
        time.sleep(0.8)

def fist(palma):
    return polegar(hand_landmarks, palma) and not finger(hand_landmarks, palma, 8) and not finger(hand_landmarks, palma, 12) and not finger(hand_landmarks, palma, 16) and not finger(hand_landmarks, palma, 20)
        


def vertical_arrow(palma, idx):
    
    label = latest_result.handedness[idx][0].category_name
    
    put_text(0, 0, 255, f"({dedos_left(palma)})", 100, 100)
    if label == "Right":
        if finger(hand_landmarks, palma, 8) and finger(hand_landmarks, palma, 12) and not finger(hand_landmarks, palma, 16) and not finger(hand_landmarks, palma, 20):
            if polegar(hand_landmarks, palma):
                pag.press('up')
            else:
                pag.press('down')
            

def finger_lines(hand_landmarks):
    for i in [4, 8, 12, 16, 20, 6, 10, 14, 18, 7, 11, 15, 19, 2, 1, 3]:
        # cx, cy = int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h)
        # cv2.line(frame, (cx, cy), (int(p_palma.x), int(p_palma.y)), (0, 0, 255), 1)
            
        x_init = int(hand_landmarks[i-1].x * w)
        y_init = int(hand_landmarks[i-1].y * h)
        x_end = int(hand_landmarks[i].x * w)
        y_end = int(hand_landmarks[i].y * h)

        cv2.line(frame, (x_init, y_init), (x_end, y_end), (255, 255, 255), 1)

    for i in [0, 5, 9, 13, 17]:
        x_init = int(hand_landmarks[i].x * w)
        y_init = int(hand_landmarks[i].y * h)
        x_end = int(hand_landmarks[i - 4].x * w)
        y_end = int(hand_landmarks[i - 4].y * h)

        cv2.line(frame, (x_init, y_init), (x_end, y_end), (255, 255, 255), 1)

# ==================================== MAIN ==============================================


cap = cv2.VideoCapture(0) # cap = webcam"

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_result and latest_result.hand_landmarks:
            

            for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):

                rp0 = hand_landmarks[0]
                rp5 = hand_landmarks[5]
                rp17 = hand_landmarks[17]

                p_palma = Dot(int((rp0.x + rp5.x + rp17.x) / 3 * w), int((rp0.y + rp5.y + rp17.y) / 3 * h))
                cv2.circle(frame, (int(p_palma.x), int(p_palma.y)), 5, (0, 0, 0), 1)

                # mouse_control(p_palma)
                # scroll_control(p_palma)
                # side_arrow(p_palma)

                put_text(0, 0, 255, f"({dedos_left(p_palma)})", 100, 100)
                finger_lines(hand_landmarks)

                # label = latest_result.handedness[idx].category_name

                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    circle_design(cx, cy)

        cv2.imshow('MediaPipe Tasks Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
