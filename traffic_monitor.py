import cv2
import time
import asyncio
import websockets
from fastai.vision.all import *

# 1. Modellarni yuklash
# 'stop_sign_model.pkl' - bu learn.export() orqali saqlangan bo'lishi kerak
stop_sign_model = load_learner('stop_sign_model.pkl')

# Vehicle model uchun ham .pkl fayl ishlatish tavsiya etiladi
# Agar sizda faqat .pth (save) bo'lsa, avval learnerni yaratib keyin load qiling
# Lekin eng yaxshisi: vehicle_model = load_learner('vehicle-model.pkl')
vehicle_model = load_learner('vehicle-model.pkl') 

# 2. Bashorat qilish funksiyasi (Eski kodda aynan shu yo'q edi)
def get_prediction(img, model):
    pred, pred_idx, probs = model.predict(img)
    return str(pred)

# 3. WebSocket xabar yuborish (Soddalashtirilgan)
async def send_event():
    uri = "ws://your-websocket-server-address"
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("Vehicle detected for more than 10 seconds")
    except Exception as e:
        print(f"WebSocket xatosi: {e}")

# 4. Video oqimini ochish
cap = cv2.VideoCapture(0)
vehicle_detected_start_time = None

print("Tizim ishga tushdi. To'xtash taqiqlangan hudud nazorat qilinmoqda...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tasvirni Fastai formatiga o'tkazish
    img = PILImage.create(frame)

    # 3.27 belgisini aniqlash
    stop_sign_pred = get_prediction(img, stop_sign_model)

    if stop_sign_pred == 'stop_sign':
        # Agar belgi bo'lsa, transportni tekshiramiz
        vehicle_pred = get_prediction(img, vehicle_model)
        
        if vehicle_pred in ['car', 'bus', 'truck']:
            if vehicle_detected_start_time is None:
                vehicle_detected_start_time = time.time()
                print("Hududda transport aniqlandi...")
            else:
                elapsed_time = time.time() - vehicle_detected_start_time
                
                # 10 soniyadan oshsa xabar berish
                if elapsed_time > 10:
                    cv2.putText(frame, "QOIDA BUZILDI: 10+ SEK!", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Qoida buzilishi! Xabar yuborilmoqda...")
                    # Muhim: real vaqtda asyncio ishlashi uchun:
                    # asyncio.get_event_loop().run_until_complete(send_event())
        else:
            vehicle_detected_start_time = None
    else:
        # Belgi ko'rinmasa vaqtni reset qilamiz
        vehicle_detected_start_time = None

    # Natijani ekranda ko'rsatish
    cv2.imshow('Smart Traffic Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
