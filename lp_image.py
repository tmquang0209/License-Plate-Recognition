from PIL import Image
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import argparse

# ===============================
# LOAD MODEL 1 LẦN (QUAN TRỌNG)
# ===============================
yolo_LP_detect = torch.hub.load(
    'yolov5',
    'custom',
    path='model/LP_detector.pt',
    source='local',
    force_reload=False
)

yolo_license_plate = torch.hub.load(
    'yolov5',
    'custom',
    path='model/LP_ocr.pt',
    source='local',
    force_reload=False
)
yolo_license_plate.conf = 0.60


# ===============================
# CORE FUNCTION DÙNG CHO FASTAPI
# ===============================
def recognize_license_plate(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image")

    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()

    list_read_plates = set()

    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            x = int(plate[0])
            y = int(plate[1])
            w = int(plate[2] - plate[0])
            h = int(plate[3] - plate[1])

            crop_img = img[y:y + h, x:x + w]

            for cc in range(2):
                for ct in range(2):
                    lp = helper.read_plate(
                        yolo_license_plate,
                        utils_rotate.deskew(crop_img, cc, ct)
                    )
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        break

    return list(list_read_plates)


# ===============================
# CHẠY CLI (KHÔNG ẢNH HƯỞNG API)
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image")
    args = parser.parse_args()

    result = recognize_license_plate(args.image)
    print("Detected license plates:", result)
