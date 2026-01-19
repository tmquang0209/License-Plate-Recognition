import os
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lp_image import recognize_license_plate

app = FastAPI()

# Cho phép CORS nếu cần
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lưu file tạm thời
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        # Gọi hàm nhận diện biển số (bạn cần chỉnh lại cho phù hợp)
        plate = recognize_license_plate(file_location)
        result = {"license_plate": plate}
    except Exception as e:
        result = {"error": str(e)}
    finally:
        os.remove(file_location)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
