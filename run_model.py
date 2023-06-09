import numpy as np
import cv2
import onnx , onnxruntime
import torch
import matplotlib.pyplot as plt
import torchvision
tfms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
])

onnx_session = onnxruntime.InferenceSession("onnxUNet.onnx")
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

video = "test_video_for_segmentation.mp4"
window_name = "Segmentation"
window_width = 800
window_height = 600
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)
class_colors = [
    (255, 0, 0),     # Class 0 (Red)
    (0, 255, 0),     # Class 1 (Green)
    (0, 0, 255),     # Class 2 (Blue)
    (255, 255, 0),   # Class 3 (Cyan)
    (255, 0, 255),   # Class 4 (Magenta)
    (0, 255, 255),   # Class 5 (Yellow)
    (128, 0, 0),     # Class 6 (Maroon)
    (0, 128, 0),     # Class 7 (Dark Green)
    (0, 0, 128),     # Class 8 (Navy)
    (128, 128, 0),   # Class 9 (Olive)
    (128, 0, 128),   # Class 10 (Purple)
    (0, 128, 128)    # Class 11 (Teal)
]
cap = cv2.VideoCapture(video )
print(cap.isOpened())
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.resize(frame, (224, 224))
    input_tensor  = tfms(frame.copy()/255.0)
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # Add batch dimension
    ort_outs = onnx_session.run([output_name], {input_name :input_tensor })
    mask = (ort_outs[0].argmax(axis = 1).squeeze(0)*255/14).astype(np.uint8)
    #mask = mask[..., None].repeat(3, -1)
    # Create an empty image for overlaying the colored masks
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Overlay each class's mask with a unique color
    for class_index in range(len(class_colors)):
        # Create a binary mask for the current class
        class_mask = np.where(mask == class_index, 255, 0).astype(np.uint8)

        # Apply the color of the current class to the overlay image
        overlay[class_mask > 0] = class_colors[class_index]

    #print(mask.shape , frame.shape , mask.dtype, frame.dtype)
    # Overlay the mask on the frame
    result = cv2.addWeighted(frame, 0.7, overlay, 0.6, 0)

    # Display the frame with the overlay
    cv2.imshow("Segmentation", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


