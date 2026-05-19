import cv2
import numpy as np
import time
import threading
from playsound import playsound
from collections import deque

# ==========================
# CONFIGURAÇÃO
# ==========================

# Região da água (ROI)
ROI_X = 100
ROI_Y = 200
ROI_W = 500
ROI_H = 250

# Sensibilidade
LIMIAR_MOVIMENTO = 1.0
AREA_MINIMA = 5000

# Histórico da IA
historico = deque(maxlen=30)

# Controle de alerta
ultimo_alerta = 0
INTERVALO_ALERTA = 20

# ==========================
# ÁUDIO
# ==========================

AUDIO_ALERTA = "ssstik.io_1778940651943.mp3"

audio_tocando = False

def tocar_audio():

    global audio_tocando

    try:

        audio_tocando = True

        playsound(AUDIO_ALERTA)

    except Exception as erro:

        print(f"Erro ao tocar áudio: {erro}")

    finally:

        audio_tocando = False

def alerta_sonoro():

    # Evita múltiplos áudios simultâneos
    if not audio_tocando:

        thread_audio = threading.Thread(
            target=tocar_audio,
            daemon=True
        )

        thread_audio.start()

# ==========================
# CÂMERA
# ==========================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

ret, frame1 = cap.read()

if not ret:
    print("Erro na câmera")
    exit()

# ROI inicial
roi1 = frame1[
    ROI_Y:ROI_Y+ROI_H,
    ROI_X:ROI_X+ROI_W
]

prvs = cv2.cvtColor(
    roi1,
    cv2.COLOR_BGR2GRAY
)

while True:

    ret, frame2 = cap.read()

    if not ret:
        break

    # ==========================
    # RECORTE DA ÁGUA
    # ==========================

    roi2 = frame2[
        ROI_Y:ROI_Y+ROI_H,
        ROI_X:ROI_X+ROI_W
    ]

    next_frame = cv2.cvtColor(
        roi2,
        cv2.COLOR_BGR2GRAY
    )

    # Redução de ruído
    next_frame = cv2.GaussianBlur(
        next_frame,
        (7,7),
        0
    )

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

    mag, ang = cv2.cartToPolar(
        flow[...,0],
        flow[...,1]
    )

    # ==========================
    # IA SIMPLES
    # ==========================

    movimento_medio = np.mean(mag)

    historico.append(movimento_medio)

    media_historica = np.mean(historico)

    confianca = 0

    # Movimento acima do normal
    if media_historica > LIMIAR_MOVIMENTO:
        confianca += 1

    # Movimento horizontal predominante
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
        230,
        cv2.THRESH_BINARY
    )

    movimento = movimento.astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)

    movimento = cv2.morphologyEx(
        movimento,
        cv2.MORPH_OPEN,
        kernel
    )

    movimento = cv2.dilate(
        movimento,
        kernel,
        iterations=2
    )

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
    # DECISÃO FINAL
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

        # Reproduz alerta a cada 20 segundos
        # enquanto continuar detectando
        if agora - ultimo_alerta > INTERVALO_ALERTA:

            print("💧 ALERTA REAL DETECTADO")

            alerta_sonoro()

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

    # ESC fecha o programa
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================
# FINALIZAÇÃO
# ==========================

cap.release()

cv2.destroyAllWindows()