import mediapipe as mp
import cv2
import time
import math

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
    cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)

def hip(dx, dy):
    return math.sqrt(math.fabs(math.pow(dx, 2) + math.pow(dy, 2)))

def polegar(hand_landmarks, palma):
    polegar_palma = hip(hand_landmarks[4].x - palma.x/w, hand_landmarks[4].y - palma.y/h)
    polegar_base = hip(hand_landmarks[4].x - hand_landmarks[2].x, hand_landmarks[4].y - hand_landmarks[2].y)
    polegar_ind_base = hip(hand_landmarks[4].x - hand_landmarks[5].x, hand_landmarks[4].y - hand_landmarks[5].y)

    return polegar_palma > polegar_base and polegar_palma > polegar_ind_base


def dedos_left(palma):
    num_dedos = 0

    if latest_result:
        for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
            label = latest_result.handedness[idx][0].category_name

            if label == "Left":
                if polegar(hand_landmarks, palma):
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
                
    return num_dedos



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

                rp0 = latest_result.hand_landmarks[idx][0]
                rp5 = latest_result.hand_landmarks[idx][5]
                rp17 = latest_result.hand_landmarks[idx][17]

                p_palma = Dot(int((rp0.x + rp5.x + rp17.x) / 3 * w), int((rp0.y + rp5.y + rp17.y) / 3 * h))
                cv2.circle(frame, (int(p_palma.x), int(p_palma.y)), 5, (0, 0, 0), 1)

                put_text(255, 255, 255, f"({dedos_left(p_palma)}))", 50, 50)
                put_text(255, 255, 255, f"({dedos_right(p_palma)}))", 50, 100)

                # label = latest_result.handedness[idx].category_name

                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    circle_design(cx, cy)

        cv2.imshow('MediaPipe Tasks Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
