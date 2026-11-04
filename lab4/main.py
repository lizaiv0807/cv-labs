import os
import cv2
import numpy as np

os.makedirs("results", exist_ok=True)

img1 = cv2.imread("images/boat1.jpg")
img2 = cv2.imread("images/boat2.jpg")

if img1 is None or img2 is None:
    raise FileNotFoundError("Не вдалося завантажити зображення з папки images/")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"Крок 1: SIFT kp1={len(kp1)}, kp2={len(kp2)}")

matcher = cv2.BFMatcher(cv2.NORM_L2)
matches_1_to_2 = matcher.knnMatch(des1, des2, k=2)
matches_2_to_1 = matcher.knnMatch(des2, des1, k=2)

print("\nКрок 3:")
T_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
for T in T_values:
    filtered = [m for m, n in matches_1_to_2 if m.distance < T * n.distance]
    print(f"  T = {T:0.2f} -> {len(filtered)}")

chosen_T = 0.75
ratio_matches_12 = [m for m, n in matches_1_to_2 if m.distance < chosen_T * n.distance]
best_2_to_1 = {m.queryIdx: m.trainIdx for m, n in matches_2_to_1 if m.distance < chosen_T * n.distance}

mutual_matches = []
for m in ratio_matches_12:
    if m.trainIdx in best_2_to_1 and best_2_to_1[m.trainIdx] == m.queryIdx:
        mutual_matches.append(m)

print(f"\nКрок 4:")
print(f"  Lowe ratio: {len(ratio_matches_12)}")
print(f"  Cross-check: {len(mutual_matches)}")

raw_matches_vis = [m[0] for m in matches_1_to_2[:100]]
vis_raw = cv2.drawMatches(
    img1, kp1, img2, kp2, raw_matches_vis, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/step5_raw_matches.jpg", vis_raw)

src_pts = np.float32([kp1[m.queryIdx].pt for m in mutual_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in mutual_matches]).reshape(-1, 1, 2)

H_ls, _ = cv2.findHomography(src_pts, dst_pts, 0)
print("\nКрок 6. H (МНК):")
print(H_ls)

h2, w2 = img2.shape[:2]
warped_ls = cv2.warpPerspective(img1, H_ls, (w2, h2))
cv2.imwrite("results/step7_warped_least_squares.jpg", warped_ls)

src_proj = cv2.perspectiveTransform(src_pts, H_ls)
reproj_errors_ls = np.linalg.norm(dst_pts - src_proj, axis=2)
mean_error_ls = np.mean(reproj_errors_ls)
print(f"Крок 7: середня похибка МНК = {mean_error_ls:.2f}")



import math
import matplotlib.pyplot as plt

def compute_reproj_error(pts_src, pts_dst, H):
    proj = cv2.perspectiveTransform(pts_src, H)
    return np.linalg.norm(pts_dst - proj, axis=2).ravel()

def custom_ransac(pts_src, pts_dst, threshold=3.0, n_iters=2000):
    best_inliers = []
    best_H = None
    num_pts = len(pts_src)
    
    for _ in range(n_iters):
        idx = np.random.choice(num_pts, 4, replace=False)
        src_sample = pts_src[idx].squeeze(1)
        dst_sample = pts_dst[idx].squeeze(1)
        
        H = cv2.getPerspectiveTransform(src_sample, dst_sample)
        if H is None or np.isnan(H).any() or np.isinf(H).any():
            continue
            
        errors = compute_reproj_error(pts_src, pts_dst, H)
        inliers = np.where(errors < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            
    if len(best_inliers) >= 4:
        best_H, _ = cv2.findHomography(pts_src[best_inliers], pts_dst[best_inliers], 0)
        
    return best_H, best_inliers

H_custom, custom_inliers = custom_ransac(src_pts, dst_pts, threshold=3.0, n_iters=2000)
custom_errors = compute_reproj_error(src_pts[custom_inliers], dst_pts[custom_inliers], H_custom)
mean_err_custom = np.mean(custom_errors)

print(f"\nКрок 8 (Власний RANSAC, tau=3.0):")
print(f"  Кількість узгоджених: {len(custom_inliers)} / {len(src_pts)}")
print(f"  Середня похибка репроєкції: {mean_err_custom:.3f} px")

inlier_ratio = len(custom_inliers) / len(src_pts)
eps = 1.0 - inlier_ratio
p = 0.99
s = 4
theoretical_N = math.log(1 - p) / math.log(1 - (1 - eps)**s)

print(f"\nКрок 9: Частка викидів eps = {eps:.3f}")
print(f"  Теоретична кількість ітерацій N = {theoretical_N:.1f}")
print(f"  Фактично використано: 2000")

H_cv, mask_cv = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
cv_inliers = np.where(mask_cv.ravel() == 1)[0]
cv_errors = compute_reproj_error(src_pts[cv_inliers], dst_pts[cv_inliers], H_cv)
mean_err_cv = np.mean(cv_errors)

print(f"\nКрок 10 (Звірка з OpenCV RANSAC):")
print(f"  OpenCV inliers: {len(cv_inliers)}, похибка: {mean_err_cv:.3f} px")
print(f"  Custom inliers: {len(custom_inliers)}, похибка: {mean_err_custom:.3f} px")

tau_values = [1, 2, 3, 5, 8, 12, 20]
inliers_counts = []
errors_at_tau = []

print("\nКрок 11. Дослідження порога tau:")
for tau in tau_values:
    H_t, mask_t = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(tau))
    inl = np.where(mask_t.ravel() == 1)[0]
    errs = compute_reproj_error(src_pts[inl], dst_pts[inl], H_t)
    m_err = np.mean(errs)
    inliers_counts.append(len(inl))
    errors_at_tau.append(m_err)
    print(f"  tau = {tau:2d} -> inliers: {len(inl):4d}, похибка: {m_err:.3f} px")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(tau_values, inliers_counts, marker='o', color='b')
ax1.set_title("Кількість узгоджених точок від tau")
ax1.set_xlabel("Порог tau (пікселі)")
ax1.set_ylabel("Кількість інлаєрів")
ax1.grid(True)

ax2.plot(tau_values, errors_at_tau, marker='s', color='r')
ax2.set_title("Середня похибка репроєкції від tau")
ax2.set_xlabel("Порог tau (пікселі)")
ax2.set_ylabel("Похибка (пікселі)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("results/step11_tau_analysis.png", dpi=300)
plt.close()
print("Крок 11: Графіки збережено в results/step11_tau_analysis.png")

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
vis_canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
vis_canvas[:h1, :w1] = img1
vis_canvas[:h2, w1:w1+w2] = img2

inlier_set = set(custom_inliers)
num_samples = min(200, len(mutual_matches))
sample_indices = np.random.choice(len(mutual_matches), num_samples, replace=False)

for idx in sample_indices:
    pt1 = tuple(np.round(kp1[mutual_matches[idx].queryIdx].pt).astype(int))
    pt2 = tuple(np.round(kp2[mutual_matches[idx].trainIdx].pt).astype(int) + np.array([w1, 0]))
    
    if idx in inlier_set:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
        
    cv2.line(vis_canvas, pt1, pt2, color, 1, cv2.LINE_AA)
    cv2.circle(vis_canvas, pt1, 3, color, -1)
    cv2.circle(vis_canvas, pt2, 3, color, -1)

cv2.imwrite("results/step12_inliers_outliers.jpg", vis_canvas)
print("Крок 12: Візуалізацію інлаєрів (зелений) та аутлаєрів (червоний) збережено в results/step12_inliers_outliers.jpg")



image_files = [f"images/boat{i}.jpg" for i in range(1, 6)]
imgs = [cv2.imread(f) for f in image_files]

kps_list = []
des_list = []
for im in imgs:
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    k, d = sift.detectAndCompute(g, None)
    kps_list.append(k)
    des_list.append(d)

pair_H = []
pair_errors = []

for i in range(len(imgs) - 1):
    m_12 = matcher.knnMatch(des_list[i], des_list[i+1], k=2)
    m_21 = matcher.knnMatch(des_list[i+1], des_list[i], k=2)
    
    rat_12 = [m for m, n in m_12 if m.distance < 0.75 * n.distance]
    b_21 = {m.queryIdx: m.trainIdx for m, n in m_21 if m.distance < 0.75 * n.distance}
    
    mut = [m for m in rat_12 if m.trainIdx in b_21 and b_21[m.trainIdx] == m.queryIdx]
    
    p_src = np.float32([kps_list[i][m.queryIdx].pt for m in mut]).reshape(-1, 1, 2)
    p_dst = np.float32([kps_list[i+1][m.trainIdx].pt for m in mut]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(p_src, p_dst, cv2.RANSAC, 3.0)
    inl = np.where(mask.ravel() == 1)[0]
    
    errs = compute_reproj_error(p_src[inl], p_dst[inl], H)
    pair_errors.append(np.mean(errs))
    pair_H.append(H)

ref_idx = 2
H_to_ref = [None] * len(imgs)
H_to_ref[ref_idx] = np.eye(3, dtype=np.float64)

# i -> i+1 => pair_H[i] переносить i в систему i+1
# pair_H[0]: 0 -> 1
# pair_H[1]: 1 -> 2
# pair_H[2]: 2 -> 3
# pair_H[3]: 3 -> 4

H_to_ref[1] = pair_H[1]
H_to_ref[0] = pair_H[1] @ pair_H[0]

inv_H2 = np.linalg.inv(pair_H[2])
inv_H3 = np.linalg.inv(pair_H[3])

H_to_ref[3] = inv_H2
H_to_ref[4] = inv_H2 @ inv_H3

all_corners = []
for i, im in enumerate(imgs):
    h, w = im.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj_corners = cv2.perspectiveTransform(corners, H_to_ref[i])
    all_corners.append(proj_corners)

all_corners = np.concatenate(all_corners, axis=0)
x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

T = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
], dtype=np.float64)

canvas_w = x_max - x_min
canvas_h = y_max - y_min

panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

for i, im in enumerate(imgs):
    H_final = T @ H_to_ref[i]
    h, w = im.shape[:2]
    
    warped_img = cv2.warpPerspective(im, H_final, (canvas_w, canvas_h))
    
    mask = (warped_img > 0).any(axis=2).astype(np.uint8)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_map = np.expand_dims(dist_map, axis=2)
    
    panorama += warped_img.astype(np.float32) * dist_map
    weight_sum += dist_map

weight_sum[weight_sum == 0] = 1.0
panorama = (panorama / weight_sum).astype(np.uint8)

cv2.imwrite("results/step15_panorama.jpg", panorama)
print("\nКрок 15: Панораму збережено в results/step15_panorama.jpg")

print(f"\nКрок 16 (Оцінка якості):")
print(f"  Середні похибки пар (px): {[round(e, 3) for e in pair_errors]}")
print(f"  Загальна середня похибка репроєкції: {np.mean(pair_errors):.3f} px")

# ==========================================
# ЧАСТИНА E. Випадок відмови (Крок 17)
# ==========================================

print("\nКрок 17. Симуляція випадку відмови (недостатнє перекриття < 15%):")
m_fail = matcher.knnMatch(des_list[0], des_list[4], k=2)
rat_fail = [m for m, n in m_fail if m.distance < 0.75 * n.distance]
print(f"  Кадри 1 і 5 (без спільної сцени): знайдено кандидатів = {len(rat_fail)}")

if len(rat_fail) < 4:
    print("  RANSAC відмовив: недостатньо точок для побудови H (< 4)")
else:
    p_src_f = np.float32([kps_list[0][m.queryIdx].pt for m in rat_fail]).reshape(-1, 1, 2)
    p_dst_f = np.float32([kps_list[4][m.trainIdx].pt for m in rat_fail]).reshape(-1, 1, 2)
    H_f, mask_f = cv2.findHomography(p_src_f, p_dst_f, cv2.RANSAC, 3.0)
    inl_f = np.sum(mask_f) if mask_f is not None else 0
    print(f"  Узгоджених точок (inliers): {inl_f} з {len(rat_fail)}")
    print("  Висновок: зіставлення систематично хибне або гомографія не має геометричного змісту через відсутність спільної площини.")