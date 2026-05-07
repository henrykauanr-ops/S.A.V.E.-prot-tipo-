import cv2
import numpy as np
import time
import webbrowser
from urllib.parse import quote
import pyautogui
from collections import deque

# ==========================
# CONFIGURAÇÃO
# ==========================

CONTATOS = [
    ("Henry", "5592985955003"),
]

MENSAGEM_PADRAO = "💧 Alerta: forte ondulação detectada no igarapé!"

# Região da água (ROI)
ROI_X = 100
ROI_Y = 200
ROI_W = 500
ROI_H = 250

# Sensibilidade
LIMIAR_MOVIMENTO = 2.0
AREA_MINIMA = 4000

# Histórico da IA
historico = deque(maxlen=30)

# Controle de alerta
ultimo_alerta = 0
INTERVALO_ALERTA = 20

# ==========================
# WHATSAPP
# ==========================

def enviar_whatsapp():

    for nome, telefone in CONTATOS:

        mensagem = f"Olá {nome}, {MENSAGEM_PADRAO}"


        link = f"https://web.whatsapp.com/send?phone={telefone}&text={quote(mensagem)}"

        webbrowser.open(link)

        time.sleep(10)

        pyautogui.press('enter')

        time.sleep(2)

        pyautogui.hotkey('ctrl', 'w')

        time.sleep(2)

        pyautogui.hotkey('enter')

# ==========================
# CÂMERA
# ==========================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

ret, frame1 = cap.read()

if not ret:
    print("Erro na câmera")
    exit()

# ROI inicial
roi1 = frame1[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

prvs = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame2 = cap.read()

    if not ret:
        break

    # ==========================
    # RECORTE DA ÁGUA
    # ==========================

    roi2 = frame2[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

    next_frame = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # Blur reduz ruído
    next_frame = cv2.GaussianBlur(next_frame, (7,7), 0)

    # ==========================
    # OPTICAL FLOW
    # ==========================

    flow = cv2.calcOpticalFlowFarneback(
        prvs,
        next_frame,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # ==========================
    # IA SIMPLES
    # ==========================

    movimento_medio = np.mean(mag)

    historico.append(movimento_medio)

    media_historica = np.mean(historico)

    # Detecta padrão contínuo de água
    confianca = 0

    if media_historica > LIMIAR_MOVIMENTO:
        confianca += 1

    # Ondulações costumam ter movimento horizontal
    horizontal = np.mean(np.abs(flow[...,0]))
    vertical = np.mean(np.abs(flow[...,1]))

    if horizontal > vertical:
        confianca += 1

    # ==========================
    # MÁSCARA
    # ==========================

    _, movimento = cv2.threshold(
        mag,
        LIMIAR_MOVIMENTO,
        255,
        cv2.THRESH_BINARY
    )

    movimento = movimento.astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)

    movimento = cv2.morphologyEx(
        movimento,
        cv2.MORPH_OPEN,
        kernel
    )

    movimento = cv2.dilate(movimento, kernel, iterations=2)

    contours, _ = cv2.findContours(
        movimento,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    deteccao = False

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < AREA_MINIMA:
            continue

        deteccao = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            roi2,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

    # ==========================
    # DECISÃO FINAL DA IA
    # ==========================

    if deteccao and confianca >= 2:

        cv2.putText(
            frame2,
            "ONDULACAO DETECTADA",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        agora = time.time()

        if agora - ultimo_alerta > INTERVALO_ALERTA:

            print("💧 ALERTA REAL DETECTADO")

            enviar_whatsapp()

            ultimo_alerta = agora

    # ==========================
    # DESENHA ROI
    # ==========================

    cv2.rectangle(
        frame2,
        (ROI_X, ROI_Y),
        (ROI_X+ROI_W, ROI_Y+ROI_H),
        (255,0,0),
        2
    )

    # ==========================
    # EXIBIÇÃO
    # ==========================

    cv2.imshow("Camera", frame2)
    cv2.imshow("Movimento", movimento)

    prvs = next_frame

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()