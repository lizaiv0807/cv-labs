import cv2
import numpy as np
from scipy.spatial import cKDTree

def custom_harris(image_gray, k=0.04, threshold_ratio=0.01, nms_radius=10, max_points=1000):
    img_float = np.float32(image_gray)
    
    Ix = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    
    Ix2_blur = cv2.GaussianBlur(Ix2, (0, 0), 1.5)
    Iy2_blur = cv2.GaussianBlur(Iy2, (0, 0), 1.5)
    IxIy_blur = cv2.GaussianBlur(IxIy, (0, 0), 1.5)
    
    det_M = (Ix2_blur * Iy2_blur) - (IxIy_blur ** 2)
    trace_M = Ix2_blur + Iy2_blur
    R = det_M - k * (trace_M ** 2)
    
    threshold = threshold_ratio * R.max()
    y_coords, x_coords = np.where(R > threshold)
    responses = R[y_coords, x_coords]
    
    sorted_indices = np.argsort(responses)[::-1]
    sorted_points = np.column_stack((x_coords[sorted_indices], y_coords[sorted_indices]))
    
    selected_points = []
    for pt in sorted_points:
        if len(selected_points) == 0:
            selected_points.append(pt)
            continue
            
        diff = np.array(selected_points) - pt
        distances_sq = np.sum(diff ** 2, axis=1)
        
        if np.min(distances_sq) > nms_radius ** 2:
            selected_points.append(pt)
            if len(selected_points) >= max_points:
                break
                
    return np.array(selected_points)

if __name__ == "__main__":
    img_path = 'data/5_bookshelf.jpg'
    img_color = cv2.imread(img_path)
    
    if img_color is None:
        print(f"Помилка: не вдалося завантажити {img_path}")
    else:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        img_gray_half = cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5)
        
        print("Шукаємо точки на оригіналі...")
        pts_orig = custom_harris(img_gray)
        
        print("Шукаємо точки на зменшеному зображенні...")
        pts_half = custom_harris(img_gray_half)
        
        pts_mapped = pts_half * 2
        
        if len(pts_orig) > 0 and len(pts_mapped) > 0:
            tree = cKDTree(pts_mapped)
            distances, _ = tree.query(pts_orig)
            
            matches = np.sum(distances < 3.0)
            repeatability = (matches / len(pts_orig)) * 100
        else:
            repeatability = 0.0
            matches = 0
            
        print("-" * 40)
        print(f"Точок на оригіналі: {len(pts_orig)}")
        print(f"Точок на зменшеній копії: {len(pts_half)}")
        print(f"Знайдено збігів: {matches}")
        print(f"Повторюваність Гарріса при зміні масштабу (0.5x): {repeatability:.2f}%")
        print("-" * 40)
        print("Висновок для звіту: Гарріс НЕ є інваріантним до масштабу (повторюваність дуже низька).")