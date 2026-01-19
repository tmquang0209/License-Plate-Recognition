
import os
import secrets
import shutil

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from lp_image import recognize_license_plate

security = HTTPBearer()
API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    API_TOKEN = secrets.token_urlsafe(32)
    os.environ["API_TOKEN"] = API_TOKEN

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

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
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
    print(f"[INFO] API_TOKEN for authentication: {API_TOKEN}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
