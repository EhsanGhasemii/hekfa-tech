import cv2
import argparse
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize face analysis model
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0)

def get_face_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    faces = face_app.get(img)
    if not faces:
        raise ValueError(f"No face detected in {img_path}")

    return faces[0].embedding  # Use first detected face only

def main():
    parser = argparse.ArgumentParser(description="Compare faces in two images.")
    parser.add_argument("image1", type=str, help="Path to first image")
    parser.add_argument("image2", type=str, help="Path to second image")
    args = parser.parse_args()

    embedding1 = get_face_embedding(args.image1)
    embedding2 = get_face_embedding(args.image2)

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"\nCosine similarity: {similarity:.4f}")

    threshold = 0.35  # Adjust based on accuracy needs
    if similarity > threshold:
        print("Same person")
    else:
        print("Different persons")

if __name__ == "__main__":
    main()
