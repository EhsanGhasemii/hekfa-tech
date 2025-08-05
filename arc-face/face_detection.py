import cv2
import insightface
from insightface.app import FaceAnalysis

# Initialize the face analysis model
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0)  # Use GPU (CUDA)

# Load image
img = cv2.imread('persons.jpg')  # Replace with your image filename
if img is None:
    raise FileNotFoundError("Image not found!")

# Detect faces
faces = face_app.get(img)

# Draw results on image
for face in faces:
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw facial landmarks
    for (x, y) in face.kps:
        cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)

# Save result image
output_path = 'output.jpg'
cv2.imwrite(output_path, img)
print(f"Output image saved as: {output_path}")
