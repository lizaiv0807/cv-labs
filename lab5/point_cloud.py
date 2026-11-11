import cv2
import numpy as np

def main():
    print("1. Завантаження зображень...")
    
    imgL_color = cv2.imread('img/im0.png', cv2.IMREAD_COLOR)
    imgL_color = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB) 
    
    imgL = cv2.cvtColor(imgL_color, cv2.COLOR_RGB2GRAY)
    imgR = cv2.imread('img/im1.png', cv2.IMREAD_GRAYSCALE)
    
    print("2. Обчислення диспаратності (SGBM)...")
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192, blockSize=15, uniquenessRatio=5,
                                 P1=8*1*15**2, P2=32*1*15**2, disp12MaxDiff=1, speckleWindowSize=100, speckleRange=32)
    disp_sgbm = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0
    
    print("3. Фільтрація поодиноких точок (Median Blur)...")

    disp_filtered = cv2.medianBlur(disp_sgbm, 5)
    
    print("4. Генерація 3D координат...")
    
    f = 3997.684
    B = 193.001
    cx = 1176.728
    cy = 1011.728
    

    Q = np.float32([
        [1, 0,  0, -cx],
        [0, 1,  0, -cy],
        [0, 0,  0,  f ],
        [0, 0, -1/B, 0]
    ])
    

    points_3D = cv2.reprojectImageTo3D(disp_filtered, Q)
    points_3D[:, :, 2] = np.abs(points_3D[:, :, 2]) 
    mask = (disp_filtered > 0) & (points_3D[:, :, 2] < 15000) 
    
    out_points = points_3D[mask]
    out_colors = imgL_color[mask]
    
    print(f"5. Збереження хмари точок у PLY (кількість точок: {len(out_points)})... Зачекай 10-15 секунд.")

    with open('cloud.ply', 'w') as f_ply:
        f_ply.write("ply\n")
        f_ply.write("format ascii 1.0\n")
        f_ply.write(f"element vertex {len(out_points)}\n")
        f_ply.write("property float x\n")
        f_ply.write("property float y\n")
        f_ply.write("property float z\n")
        f_ply.write("property uchar red\n")
        f_ply.write("property uchar green\n")
        f_ply.write("property uchar blue\n")
        f_ply.write("end_header\n")
        for i in range(len(out_points)):
            p = out_points[i]
            c = out_colors[i]

            f_ply.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
            
    print("\nГотово! Файл 'cloud.ply' з'явився у твоїй папці lab5.")

if __name__ == "__main__":
    main()