import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import glob

def nms(points, responses, radius=10, max_points=1000):
    if len(points) == 0:
        return np.array([])
    sorted_indices = np.argsort(responses)[::-1]
    sorted_points = points[sorted_indices]
    
    selected_points = []
    for pt in sorted_points:
        if not selected_points:
            selected_points.append(pt)
            continue
        diff = np.array(selected_points) - pt
        if np.min(np.sum(diff ** 2, axis=1)) > radius ** 2:
            selected_points.append(pt)
            if len(selected_points) >= max_points:
                break
    return np.array(selected_points)

def detect_harris(img_gray):
    img_float = np.float32(img_gray)
    Ix = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    Ix2 = cv2.GaussianBlur(Ix * Ix, (0, 0), 1.5)
    Iy2 = cv2.GaussianBlur(Iy * Iy, (0, 0), 1.5)
    IxIy = cv2.GaussianBlur(Ix * Iy, (0, 0), 1.5)
    R = (Ix2 * Iy2) - (IxIy ** 2) - 0.04 * ((Ix2 + Iy2) ** 2)
    y, x = np.where(R > 0.01 * R.max())
    if len(x) == 0: return np.array([])
    return nms(np.column_stack((x, y)), R[y, x])

def detect_sift(img_gray):
    sift = cv2.SIFT_create(nfeatures=5000)
    kp = sift.detect(img_gray, None)
    if not kp: return np.array([])
    return nms(np.array([k.pt for k in kp]), np.array([k.response for k in kp]))

def detect_orb(img_gray):
    orb = cv2.ORB_create(nfeatures=5000)
    kp = orb.detect(img_gray, None)
    if not kp: return np.array([])
    return nms(np.array([k.pt for k in kp]), np.array([k.response for k in kp]))

def apply_transform(img, t_type, val):
    h, w = img.shape[:2]
    if t_type == 'rot':
        M = cv2.getRotationMatrix2D((w//2, h//2), val, 1.0)
        H = np.vstack([M, [0, 0, 1]])
        return cv2.warpPerspective(img, H, (w, h)), H
    elif t_type == 'scale':
        H = np.array([[val, 0, 0], [0, val, 0], [0, 0, 1]], dtype=float)
        return cv2.warpPerspective(img, H, (int(w*val), int(h*val))), H
    elif t_type == 'bright':
        return np.clip(img.astype(float) * val, 0, 255).astype(np.uint8), np.eye(3)
    elif t_type == 'noise':
        noise = np.random.normal(0, val, img.shape)
        return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8), np.eye(3)

def repeatability(pts1, pts2, H, eps=3):
    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts1_m = (H @ pts1_h.T).T
    pts1_m = pts1_m[:, :2] / pts1_m[:, 2:]
    
    tree = cKDTree(pts2)
    dist, _ = tree.query(pts1_m)
    return np.sum(dist <= eps) / min(len(pts1), len(pts2))

if __name__ == "__main__":
    os.makedirs('figures', exist_ok=True)
    
    image_paths = glob.glob('data/*.jpg')
    if not image_paths:
        print("Помилка: Не знайдено зображень у папці data/")
        exit()
        
    experiments = {
        'rot': {'range': range(0, 91, 10), 'title': 'Поворот (градуси)', 'file': 'rep_rot.png'},
        'scale': {'range': np.arange(0.5, 2.25, 0.25), 'title': 'Масштаб', 'file': 'rep_scale.png'},
        'bright': {'range': np.arange(0.5, 1.6, 0.1), 'title': 'Яскравість', 'file': 'rep_bright.png'},
        'noise': {'range': range(0, 21, 5), 'title': 'Гауссів шум (sigma)', 'file': 'rep_noise.png'}
    }
    
    detectors = {'Harris': detect_harris, 'SIFT': detect_sift, 'ORB': detect_orb}
    
    for exp_key, exp_data in experiments.items():
        print(f"Запуск експерименту: {exp_data['title']}...")
        x_vals = list(exp_data['range'])
        results = {det: np.zeros(len(x_vals)) for det in detectors}
        
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            pts1 = {name: func(img) for name, func in detectors.items()}
            
            for i, val in enumerate(x_vals):
                trans_img, H = apply_transform(img, exp_key, val)
                for name, func in detectors.items():
                    pts2 = func(trans_img)
                    rep = repeatability(pts1[name], pts2, H)
                    results[name][i] += rep
                    
        num_imgs = len(image_paths)
        plt.figure(figsize=(10, 6))
        for name in detectors:
            avg_rep = (results[name] / num_imgs) * 100
            plt.plot(x_vals, avg_rep, marker='o', label=name)
            
        plt.title(f"Повторюваність детекторів: {exp_data['title']} (Усереднено для {num_imgs} зображень)")
        plt.xlabel(exp_data['title'])
        plt.ylabel("Повторюваність (%)")
        plt.ylim(0, 105)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"figures/{exp_data['file']}", bbox_inches='tight')
        plt.close()
        
    print("Всі експерименти завершено! Графіки збережено у папку figures/.")