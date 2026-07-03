import cv2
import numpy as np
from app.input_handler import load_image

raw_a = load_image('v1-new.pdf', dpi=300)
raw_b = load_image('v2-new.pdf', dpi=300)

print(f"Raw A: {raw_a.shape}")
print(f"Raw B: {raw_b.shape}")

h_a, w_a = raw_a.shape[:2]
h_b, w_b = raw_b.shape[:2]

if h_a * w_a > h_b * w_b:
    scale = min(w_b / w_a, h_b / h_a)
    new_w, new_h = int(w_a * scale), int(h_a * scale)
    img_a = cv2.resize(raw_a, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_b = raw_b
else:
    scale = min(w_a / w_b, h_a / h_b)
    new_w, new_h = int(w_b * scale), int(h_b * scale)
    img_b = cv2.resize(raw_b, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_a = raw_a

print(f"Scaled A: {img_a.shape}")
print(f"Scaled B: {img_b.shape}")

target_h = max(img_a.shape[0], img_b.shape[0])
target_w = max(img_a.shape[1], img_b.shape[1])

def pad_image(img, tw, th):
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img, 0, th - h, 0, tw - w, cv2.BORDER_CONSTANT, value=[255,255,255])
    
img_a = pad_image(img_a, target_w, target_h)
img_b = pad_image(img_b, target_w, target_h)

print(f"Padded A: {img_a.shape}")
print(f"Padded B: {img_b.shape}")

# Test align
gray_ref = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_cmp = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=5000)
kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
kp_cmp, desc_cmp = orb.detectAndCompute(gray_cmp, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
raw_matches = bf.knnMatch(desc_cmp, desc_ref, k=2)

good_matches = []
for pair in raw_matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"Good matches: {len(good_matches)}")
if len(good_matches) > 10:
    pts_cmp = np.float32([kp_cmp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_cmp, pts_ref, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"Inliers: {inliers}")
