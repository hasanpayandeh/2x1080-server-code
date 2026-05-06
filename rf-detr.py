import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import supervision as sv
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRMedium()

app = FastAPI(title="RF-DETR API", version="1.0.0")


class DetectRequest(BaseModel):
	image_url: str = Field(..., min_length=1)
	threshold: float = Field(default=0.5, ge=0.0, le=1.0)
	save_annotated: bool = False
	output_path: str = "annotated.jpg"


@app.get("/health")
def health() -> dict:
	return {
		"status": "ok",
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
	}


@app.post("/detect")
def detect(request: DetectRequest) -> dict:
	image_url = request.image_url.strip()
	if not image_url:
		raise HTTPException(status_code=400, detail="image_url cannot be empty")

	detections = model.predict(image_url, threshold=request.threshold)
	labels = [f"{COCO_CLASSES[int(class_id)]}" for class_id in detections.class_id]

	results = []
	for idx, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
		detection_row = {
			"class_id": int(class_id),
			"class_name": COCO_CLASSES[int(class_id)],
			"confidence": float(confidence),
		}

		if hasattr(detections, "xyxy") and len(detections.xyxy) > idx:
			detection_row["bbox_xyxy"] = [float(v) for v in detections.xyxy[idx]]

		results.append(detection_row)

	response = {
		"count": len(detections),
		"labels": labels,
		"detections": results,
	}

	if request.save_annotated:
		annotated_image = sv.BoxAnnotator().annotate(
			detections.metadata["source_image"], detections
		)
		annotated_image = sv.LabelAnnotator().annotate(
			annotated_image, detections, labels
		)
		cv2.imwrite(request.output_path, annotated_image)
		response["annotated_image_path"] = request.output_path

	return response


if __name__ == "__main__":
	host = os.environ.get("HOST", "0.0.0.0")
	port = int(os.environ.get("PORT", "8000"))
	uvicorn.run(app, host=host, port=port)