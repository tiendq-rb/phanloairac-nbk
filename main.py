from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI(title="AI Waste Classification - THCS Nguyễn Bỉnh Khiêm")


# Tải mô hình YOLO đã huấn luyện từ thư mục runs
model = YOLO("best.pt")

# Từ điển hướng dẫn xử lý rác theo từng lớp (Bạn đổi tên class cho khớp với lúc train trên Roboflow)
DISPOSAL_GUIDE = {
    "organic": "Rác hữu cơ -> Bỏ vào thùng rác màu Xanh lá để làm phân compost.",
    "recyclable": "Rác tái chế -> Làm sạch, khô và bỏ vào thùng màu Vàng/Xanh dương.",
    "non-recyclable": "Rác vô cơ khác -> Bỏ vào thùng rác sinh hoạt thông thường.",
    "hazardous": "Rác nguy hại -> Mang đến điểm thu gom nguy hại của trường/địa phương."
}

templates = Jinja2Templates(directory=".")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ client gửi lên
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    img_np = np.array(image)
    
    # Dự đoán bằng mô hình YOLO
    results = model(img_np)
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            
            # Lấy hướng dẫn xử lý
            guide = DISPOSAL_GUIDE.get(cls_name, "Bỏ vào đúng khu vực quy định.")
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf * 100, 1),
                "guide": guide
            })
            
    return {"success": True, "detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
