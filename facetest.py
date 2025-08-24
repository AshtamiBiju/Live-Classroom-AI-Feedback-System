import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize detector
detector = MTCNN()

# Simple tracker: keeps a list of previous face centers
tracked_faces = []
DIST_THRESHOLD = 50  # pixels

# Function to check if a face is already tracked
def is_tracked(face_center):
    for i, center in enumerate(tracked_faces):
        dist = np.linalg.norm(np.array(face_center) - np.array(center))
        if dist < DIST_THRESHOLD:
            return i  # Return index of existing face
    return -1

# List to store cropped faces as numpy arrays
cropped_faces_array = []

# Simulate processing multiple images (replace this with your loop or camera frames)
image_paths = ["hello.jpg", "imag2.jpg", "jjj.jpg"]  # replace with your images
face_id_counter = 1

for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image {image_path} not found.")
        continue

    results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    for result in results:
        x, y, width, height = result['box']
        x, y = max(0, x), max(0, y)
        face_center = (x + width // 2, y + height // 2)
        
        idx = is_tracked(face_center)
        if idx == -1:
            # New face
            tracked_faces.append(face_center)
            face_id = face_id_counter
            face_id_counter += 1
        else:
            # Already tracked face
            face_id = idx + 1
            tracked_faces[idx] = face_center  # Update center

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.putText(img, f"ID {face_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Crop face
        face_crop = img[y:y + height, x:x + width]

        # Save face crop as image (optional)
        cv2.imwrite(f"face_{face_id}.jpg", face_crop)

        # Resize face crop to same size (optional but recommended for arrays)
        face_resized = cv2.resize(face_crop, (160, 160))
        cropped_faces_array.append(face_resized)

    print(f"Processed {image_path}, detected {len(results)} faces.")

    cv2.imshow("Faces Detected", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Convert list to numpy array
cropped_faces_array = np.array(cropped_faces_array)
print(f"Shape of cropped faces array: {cropped_faces_array.shape}")
