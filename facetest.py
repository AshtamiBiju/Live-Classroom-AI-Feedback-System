import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# --- SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Face detector
mtcnn = MTCNN(keep_all=True, device=device)

# Face embedding model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Threshold for considering a new face vs existing
SIMILARITY_THRESHOLD = 0.6

# Known faces: {id: embedding_vector}
known_faces = {}
next_face_id = 0

# Function to compute embedding for a cropped face
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = torch.tensor(face_img, device=device).permute(2, 0, 1).float() / 255.0
    face_img = face_img.unsqueeze(0)
    embedding = resnet(face_img).detach().cpu().numpy()
    return embedding[0]

# Function to find best match
def find_match(embedding):
    if not known_faces:
        return None
    min_distance = float('inf')
    matched_id = None
    for face_id, known_embedding in known_faces.items():
        dist = cosine(embedding, known_embedding)
        if dist < min_distance:
            min_distance = dist
            matched_id = face_id
    if min_distance < SIMILARITY_THRESHOLD:
        return matched_id
    return None

# --- START CAMERA ---
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]

            # Crop face region
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Get embedding
            embedding = get_embedding(face)

            # Check if this face matches an existing one
            face_id = find_match(embedding)

            # If not found, assign a new ID
            if face_id is None:
                face_id = next_face_id
                known_faces[face_id] = embedding
                next_face_id += 1

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {face_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save embeddings for later use
np.save('known_faces_embeddings.npy', known_faces)
print("Saved known faces to known_faces_embeddings.npy")
