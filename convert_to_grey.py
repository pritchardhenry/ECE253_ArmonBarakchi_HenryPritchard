import os
import cv2

FOLDER = os.path.abspath("50photosUnderexposed")

print("Working in:", FOLDER)
print("Files detected:")

files = os.listdir(FOLDER)
print(files)

for fname in files:
    if fname.lower().endswith((".jpg", ".jpeg", ".JPG")):
        path = os.path.join(FOLDER, fname)
        print(f"Processing: {path}")

        img = cv2.imread(path)
        if img is None:
            print(f"  ERROR: OpenCV could not read {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        success = cv2.imwrite(path, gray)
        if success:
            print(f"  Converted → {fname}")
        else:
            print(f"  ERROR: Could not write {fname}")
