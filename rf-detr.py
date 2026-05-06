import supervision as sv
import cv2
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRMedium()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

print(f"Detections: {len(detections)}")
for class_id, confidence in zip(detections.class_id, detections.confidence):
	print(f"{COCO_CLASSES[int(class_id)]}: {float(confidence):.3f}")

cv2.imwrite("annotated.jpg", annotated_image)
print("Saved annotated image to annotated.jpg")