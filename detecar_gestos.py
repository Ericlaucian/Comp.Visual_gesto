import mediapipe as mp
import cv2
import time
import math

# ============== Configurações da API do MediaPipe ================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
global last_distance 
last_distance = 0
radius = 10

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

def hipotenusa(dx, dy):
    return math.sqrt(dx * dx + dy * dy)


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


def linha_idicador_indicador(indicadores):
    for idx, landmarks in enumerate(latest_result.hand_landmarks):
        label = latest_result.handedness[idx][0].category_name

        ponto8 = landmarks[8]
        px, py = int(ponto8.x * w), int(ponto8.y * h)
        indicadores[label] = (px, py)
                                        
    if "Left" in indicadores and "Right" in indicadores:
        p1 = indicadores["Left"]
        p2 = indicadores["Right"]

        cv2.line(frame, p1, p2, (255, 0, 0), 3)
        put_text(0, 255, 0, f"Line: ({hipotenusa(p1[0]-p2[0], p1[1]-p2[1]):.2f})", (p1[0]+p2[0])//2, (p1[1]+p2[1])//2)




def zoom_in_out(p1, p2, last_distance, radius):
    zoom = 2
    distance = hipotenusa(abs(p1.x-p2.x), abs(p1.y-p2.y))
    if abs(distance - last_distance) > 0.02:
        if distance > last_distance:
            put_text(0, 255, 0, "Zoom In", 50, 50)
            radius += zoom
        elif distance < last_distance:
            put_text(255, 0, 0, "Zoom Out", 50, 50)
            radius -= zoom
    if radius < 10: radius = 10
    # cv2.circle(frame, (100, 100), radius, (0, 255, 0), -1)
    return (distance, radius)
    #print(last_distance)



def linha_polegar_indicador():
    if landmark in [indicador, polegar]:
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
    else:
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cx_pol, cy_pol = int(polegar.x * w), int(polegar.y * h)
    cx_ind, cy_ind = int(indicador.x * w), int(indicador.y * h)

    line_size = hipotenusa(abs(cx_ind - cx_pol), abs(cy_ind - cy_pol))

    if line_size < 40:
        cv2.line(frame, (cx_pol, cy_pol), (cx_ind, cy_ind), (0, 0, 255), 3)
    else:
        cv2.line(frame, (cx_pol, cy_pol), (cx_ind, cy_ind), (255, 0, 0), 3)

    put_text(255, 0, 0, f"Line: ({line_size:.2f})", (cx_pol + cx_ind) // 2, (cy_pol + cy_ind) // 2)
        

def dedos_estendidos():
    dedos = [4, 8, 12, 16, 20]
    num_dedos = 0
    for idx, landmarks in enumerate(latest_result.hand_landmarks):
        label = latest_result.handedness[idx][0].category_name
        for dedo in dedos:
            # num_dedos += 1

            if dedo == 4:
                if label == "Left":
                    if landmarks[dedo].x < landmarks[dedo - 2].x:
                        num_dedos += 1
                        put_text(255, 0, 0, f"{dedo//4}", 500, 50 + (dedo//4)*30)
                else:
                    if landmarks[dedo].x > landmarks[dedo - 2].x:
                        num_dedos += 1
                        put_text(0, 255, 0, f"{dedo//4}", 50, 50 + (dedo//4)*30)
            else:
                if landmarks[dedo].y < landmarks[dedo - 2].y:
                    num_dedos += 1
                    if label == "Left":
                        put_text(255, 0, 0, f"{dedo//4}", 500, 50 + (dedo//4)*30)
                    else:
                        put_text(0, 255, 0, f"{dedo//4}", 50, 50 + (dedo//4)*30)

        # put_text(255, 0, 0, f"Fingers: {num_dedos}", 50, 50)
    return num_dedos    

def dedos_left():
    dedos = [4, 8, 12, 16, 20]
    num_dedos = 0
    for idx, landmarks in enumerate(latest_result.hand_landmarks):
        label = latest_result.handedness[idx][0].category_name
        if label == "Left":
            for dedo in dedos:
                if dedo == 4:
                    if landmarks[dedo].x < landmarks[dedo - 2].x:
                        num_dedos += 1
                        put_text(255, 0, 0, f"{dedo//4}", 500, 50 + (dedo//4)*30)
                else:
                    if landmarks[dedo].y < landmarks[dedo - 2].y:
                        num_dedos += 1
                        put_text(255, 0, 0, f"{dedo//4}", 500, 50 + (dedo//4)*30)

    return num_dedos

def dedos_right():
    dedos = [4, 8, 12, 16, 20]
    num_dedos = 0
    for idx, landmarks in enumerate(latest_result.hand_landmarks):
        label = latest_result.handedness[idx][0].category_name
        if label == "Right":
            for dedo in dedos:
                if dedo == 4:
                    if landmarks[dedo].x > landmarks[dedo - 2].x:
                        num_dedos += 1
                        put_text(0, 255, 0, f"{dedo//4}", 50, 50 + (dedo//4)*30)
                else:
                    if landmarks[dedo].y < landmarks[dedo - 2].y:
                        num_dedos += 1
                        put_text(0, 255, 0, f"{dedo//4}", 50, 50 + (dedo//4)*30)

    return num_dedos

def ball_design():
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    

cap = cv2.VideoCapture(0) # cap = webcam"

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        indicadores = {}

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:

                indicador = hand_landmarks[8]
                polegar = hand_landmarks[4]
                ponto_5 = hand_landmarks[5]



                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)

                    ball_design()
                    num_dedos = dedos_left()
                    put_text(255, 255, 255, f"({num_dedos})", 50, 50)

                    if num_dedos == 0:
                        put_text(255, 0, 0, "Fist", 50, 100)

                    # if abs(polegar.x - ponto_5.x) < 0.02 and abs(polegar.x - ponto_5.x) > -0.02:
                    #     linha_idicador_indicador(indicadores)
                    # else:
                    #     linha_polegar_indicador()

                    # zoom_result = zoom_in_out(indicador, polegar, last_distance, radius)
                    # last_distance = zoom_result[0]
                    # radius = zoom_result[1]
                    # print(f"Distance: {last_distance:.2f}")

                

        cv2.imshow('MediaPipe Tasks Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
