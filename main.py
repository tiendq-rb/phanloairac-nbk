from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Tải mô hình YOLOv8
model = YOLO("best.pt")

# Từ điển hướng dẫn phân loại
waste_info = {
    "organic": "Rác hữu cơ -> Bỏ vào thùng rác màu xanh lá.",
    "recyclable": "Rác tái chế -> Làm sạch, khô và bỏ vào thùng màu vàng/xanh dương.",
    "non-recyclable": "Rác vô cơ khác -> Bỏ vào thùng rác sinh hoạt thông thường.",
    "hazardous": "Rác nguy hại -> Mang đến điểm thu gom rác nguy hại của trường."
}

# Route trang chủ: Trả về file index.html trực tiếp
@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("index.html")

# Route nhận diện rác thải
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Dự đoán bằng mô hình YOLOv8
    results = model(img)
    
    predictions = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])
            guide = waste_info.get(label.lower(), "Bỏ vào đúng khu vực quy định.")
            
            predictions.append({
                "class": label,
                "confidence": round(confidence * 100, 1),
                "guide": guide
            })

    if not predictions:
        return {"status": "success", "predictions": [{"class": "Không xác định", "confidence": 0, "guide": "Vui lòng đưa rác lại gần camera hơn."}]}

    return {"status": "success", "predictions": predictions}
