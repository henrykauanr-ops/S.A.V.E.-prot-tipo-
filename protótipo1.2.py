import cv2
import requests
import time

TOKEN = "8703875163:AAGf9nQFtWpLxS5R8OtBH8Q0bGIIufBQS_Y"
CHAT_ID = "8755626101"

def enviar_alerta():
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": "💧 Possível água detectada!"
    })

def enviar_imagem(caminho):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    
    with open(caminho, "rb") as imagem:
        requests.post(url, data={"chat_id": CHAT_ID}, files={"photo": imagem})

enviar_imagem("Captura de tela 2026-04-16 084007.png")

cap = cv2.VideoCapture(1)  # Use a webcam padrão

ret, frame1 = cap.read()
ret, frame2 = cap.read()

ultimo_alerta = 0
intervalo = 15  # segundos entre alertas

while True:
    # =====================
    # MOVIMENTO
    # =====================
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, movimento = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # =====================
    # COR (AZUL + NEUTRO)
    # =====================
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    azul_mask = cv2.inRange(hsv, (90, 50, 50), (140, 255, 255))
    neutro_mask = cv2.inRange(hsv, (0, 0, 50), (180, 50, 255))  # cinza/branco

    cor_mask = cv2.bitwise_or(azul_mask, neutro_mask)

    # =====================
    # BRILHO (REFLEXO)
    # =====================
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    _, brilho = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)

    # =====================
    # COMBINAÇÃO INTELIGENTE
    # =====================
    combinado = cv2.bitwise_and(movimento, cor_mask)
    combinado = cv2.bitwise_and(combinado, brilho)

    # Limpar ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    combinado = cv2.morphologyEx(combinado, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combinado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 20000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

        agora = time.time()
        if agora - ultimo_alerta > intervalo:
            enviar_alerta()
            ultimo_alerta = agora
            print("Água mais provável detectada!")

    cv2.imshow("Camera", frame1)
    cv2.imshow("Movimento", movimento)
    cv2.imshow("Cor", cor_mask)
    cv2.imshow("Brilho", brilho)
    cv2.imshow("Final", combinado)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()