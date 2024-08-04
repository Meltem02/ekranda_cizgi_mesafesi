import cv2
import mediapipe as mp

# Kamera bağlantısını açın
cam = cv2.VideoCapture(0)

# Video kaydı için ayarlar
fourcc = cv2.VideoWriter_fourcc(*'XVID')
width, height = 640, 480
out = cv2.VideoWriter("dronegri.avi", fourcc, 30.0, (width, height))

# Mediapipe el tespiti modülü
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    # Ortadaki çizgiyi çizin
    frame = cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 5)

    # Görüntüyü RGB formatına dönüştürün (Mediapipe için gereklidir)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El tespiti işlemini gerçekleştirin
    hlms = hands.process(imgRGB)
    
    # El tespiti başarılıysa
    if hlms.multi_hand_landmarks:
        for handlanmarks in hlms.multi_hand_landmarks:
            for fingerNum, landmark in enumerate(handlanmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)
                
                # İşaret parmağı konumu
                if fingerNum == 8:
                    cv2.circle(frame, (positionX, positionY), 30, (255, 255, 255), cv2.FILLED)

                    # İşaret parmağının çizgiye olan uzaklığını hesaplayın
                    distance_to_middle_line = abs(landmark.x * width - width // 2)

                    # Ekran üzerinde uzaklığı gösterin
                    cv2.putText(frame, f'Distance to middle line: {distance_to_middle_line}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # El üzerindeki çizgileri çizin
            mpDraw.draw_landmarks(frame, handlanmarks, mpHands.HAND_CONNECTIONS)

    # Videoyu kaydedin ve gösterin
    out.write(frame)
    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video oynatma sonlandırıldı.")
        break

# Kaynakları serbest bırakın
cam.release()
out.release()
cv2.destroyAllWindows()
