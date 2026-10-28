import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

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
                
    return np.array(selected_points), R

def verify_harris(image_gray, custom_points):
    lib_R = cv2.cornerHarris(np.float32(image_gray), blockSize=3, ksize=3, k=0.04)
    lib_threshold = 0.01 * lib_R.max()
    y_lib, x_lib = np.where(lib_R > lib_threshold)
    lib_points = np.column_stack((x_lib, y_lib))
    
    if len(lib_points) == 0 or len(custom_points) == 0:
        return 0.0
        
    tree = cKDTree(lib_points)
    distances, _ = tree.query(custom_points)
    matches = np.sum(distances < 2.0)
    
    return matches / len(custom_points)

if __name__ == "__main__":
    # Створюємо папки, якщо їх ще немає
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    img_path = 'data/5_bookshelf.jpg'
    img_color = cv2.imread(img_path)
    
    if img_color is None:
        print(f"Помилка: не вдалося завантажити зображення {img_path}")
    else:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        custom_pts, R_response = custom_harris(img_gray)
        match_percent = verify_harris(img_gray, custom_pts) * 100
        
        print(f"Знайдено точок після NMS: {len(custom_pts)}")
        print(f"Збіг із бібліотечною функцією: {match_percent:.2f}%")
        
     
        img_draw = img_color.copy()
        for pt in custom_pts:
            cv2.circle(img_draw, tuple(pt), 3, (0, 255, 0), -1)
        cv2.imwrite('results/harris_points_bookshelf.jpg', img_draw)
            
  
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
        plt.title(f"Власний Гарріс: {len(custom_pts)} точок\nЗбіг: {match_percent:.2f}%")
        plt.axis('off')
        plt.savefig('figures/part_a_plot.png', bbox_inches='tight')
        plt.show()