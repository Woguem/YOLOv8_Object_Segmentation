"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a GAN model to generate image

"""


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime

start_time = datetime.now()  # Start timer


def train_segmentation_model():
    """Train YOLOv8 segmentation model"""
    print("Starting YOLOv8 segmentation training")
    model = YOLO("yolov8n-seg.pt")
    
    results = model.train(
        data="coco128-seg.yaml",
        epochs=10,
        imgsz=640,
        batch=4,
        device="cpu",
        name='yolov8_seg',
        overlap_mask=True,
        mask_ratio=4,
    )
    print("Training completed")
    return "./runs/segment/yolov8_seg/weights/best.pt"

def load_image(image_path):
    """Load image from web or local path"""
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        return cv2.cvtColor(np.array(Image.open(BytesIO(response.content))), cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)

def run_segmentation(model_path, image_path):
    """Run segmentation on image"""
    print(f"Segmenting {image_path}")
    img = load_image(image_path)
    model = YOLO(model_path)
    
    results = model(img, conf=0.5, iou=0.45, retina_masks=True)
    
    for r in results:
        if r.masks is not None:
            # Create overlay visualization
            overlay = img.copy()
            for mask in r.masks.data.cpu().numpy():
                contours = cv2.findContours(
                    (mask * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )[0]
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            
            # Display results
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Segmentation Results")
            #plt.show()
            
            # Save outputs
            cv2.imwrite("segmentation_output.jpg", overlay)
            print("Results saved to segmentation_output.jpg")
        else:
            print("No objects detected")

def video_segmentation(model_path, video_source=None):
    """
    Run segmentation on video/webcam
    Args:
        video_source: None for webcam, 0 for default cam, or video path
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0 if video_source is None else video_source)
    
    if not cap.isOpened():
        print("Error opening video source")
        return
    
    print("Starting video segmentation (Press 'q' to quit)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, stream=True)
        for r in results:
            if r.masks is not None:
                for mask in r.masks:
                    contours = mask.xy[0].astype(np.int32)
                    cv2.polylines(frame, [contours], True, (0, 255, 0), 2)
        
        cv2.imshow("Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Example usage
trained_model = train_segmentation_model()  
test_image = "https://ultralytics.com/images/zidane.jpg"

# Image segmentation
run_segmentation(trained_model, test_image)

# Video segmentation (None for webcam)
# video_segmentation(trained_model, video_source=None)  # Use webcam
# video_segmentation(trained_model, "video.mp4")  # Or video file


end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")




