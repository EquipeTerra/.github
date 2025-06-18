from ultralytics import YOLO
import cv2
import numpy as np
import pyautogui
import time
from collections import deque

# Modelo
model = YOLO('best.pt')

# Webcam
cap = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False

# Delay de clique
last_press_time = 0
press_interval = 0.5

# Resolução da tela
screen_w, screen_h = pyautogui.size()

# Fila para suavizar o movimento
smooth_queue = deque(maxlen=5)

def is_finger_extended(tip, pip, mcp):
    tip = np.array(tip[:2])
    pip = np.array(pip[:2])
    mcp = np.array(mcp[:2])
    return np.linalg.norm(tip - mcp) > np.linalg.norm(pip - mcp)

def is_hand_open(keypoints):
    fingers = {
        'indicador': (8, 6, 5),
        'medio': (12, 10, 9),
        'anelar': (16, 14, 13),
        'mindinho': (20, 18, 17)
    }
    extended = 0
    for _, (tip, pip, mcp) in fingers.items():
        if is_finger_extended(keypoints[tip], keypoints[pip], keypoints[mcp]):
            extended += 1
    return extended >= 4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    results = model.predict(source=frame, show=False, save=False, stream=True)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            for kps in r.keypoints.data:
                keypoints = kps.cpu().numpy().tolist()
                if len(keypoints) >= 21:
                    hand_open = is_hand_open(keypoints)
                    label = "Aberta" if hand_open else "Fechada"
                    coords = tuple(int(x) for x in keypoints[0][:2])
                    cv2.putText(frame, f'mao: {label}', coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    # Posição do dedo indicador (ponto 8)
                    finger_tip = keypoints[8]
                    x, y = finger_tip[:2]

                    # Adiciona na fila
                    smooth_queue.append((x, y))

                    # Calcula média das últimas posições
                    avg_x = sum(p[0] for p in smooth_queue) / len(smooth_queue)
                    avg_y = sum(p[1] for p in smooth_queue) / len(smooth_queue)

                    # Inversão horizontal (espelhamento)
                    screen_x = int(((w - avg_x) / w) * screen_w)
                    screen_y = int((avg_y / h) * screen_h)

                    # Mover mouse
                    pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                    # Clique se mão fechada
                    if not hand_open:
                        now = time.time()
                        if now - last_press_time > press_interval:
                            pyautogui.click()
                            last_press_time = now

    cv2.imshow('Hand Pose Detection', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
