import os
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import supervision as sv
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
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


def _predict_and_build_response(
	image_source: str,
	threshold: float,
	save_annotated: bool,
	output_path: str,
) -> dict:
	detections = model.predict(image_source, threshold=threshold)
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

	if save_annotated:
		annotated_image = sv.BoxAnnotator().annotate(
			detections.metadata["source_image"], detections
		)
		annotated_image = sv.LabelAnnotator().annotate(
			annotated_image, detections, labels
		)
		cv2.imwrite(output_path, annotated_image)
		response["annotated_image_path"] = output_path

	return response


@app.post("/detect")
def detect(request: DetectRequest) -> dict:
	image_url = request.image_url.strip()
	if not image_url:
		raise HTTPException(status_code=400, detail="image_url cannot be empty")

	return _predict_and_build_response(
		image_source=image_url,
		threshold=request.threshold,
		save_annotated=request.save_annotated,
		output_path=request.output_path,
	)


@app.post("/detect_upload")
async def detect_upload(
	request: Request,
	threshold: float = Query(default=0.5, ge=0.0, le=1.0),
	save_annotated: bool = Query(default=False),
	output_path: str = Query(default="annotated.jpg"),
) -> dict:
	content_type = request.headers.get("content-type", "")
	if content_type and not content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="content-type must be an image media type")

	image_bytes = await request.body()
	if not image_bytes:
		raise HTTPException(status_code=400, detail="uploaded image file is empty")

	extension = ""
	if "/" in content_type:
		extension = content_type.split("/")[-1].split(";")[0].strip()
	suffix = f".{extension}" if extension else ".jpg"
	temp_path = ""

	try:
		with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
			temp_file.write(image_bytes)
			temp_path = temp_file.name

		return _predict_and_build_response(
			image_source=temp_path,
			threshold=threshold,
			save_annotated=save_annotated,
			output_path=output_path,
		)
	finally:
		if temp_path and os.path.exists(temp_path):
			os.remove(temp_path)


if __name__ == "__main__":
	host = os.environ.get("HOST", "0.0.0.0")
	port = int(os.environ.get("PORT", "8000"))
	uvicorn.run(app, host=host, port=port)