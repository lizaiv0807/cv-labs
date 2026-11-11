import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    imgL_color = cv2.imread('img/im0.png', cv2.IMREAD_COLOR)
    imgL_color = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB) 
    imgL = cv2.cvtColor(imgL_color, cv2.COLOR_RGB2GRAY)
    imgR = cv2.imread('img/im1.png', cv2.IMREAD_GRAYSCALE)
    
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192, blockSize=15, uniquenessRatio=5,
                                 P1=8*1*15**2, P2=32*1*15**2, disp12MaxDiff=1, speckleWindowSize=100, speckleRange=32)
    disp_sgbm = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0
    
    disp_filtered = cv2.medianBlur(disp_sgbm, 5)
    
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
    
    # Залишаємо мотоцикл і відрізаємо далеку стіну
    mask = (disp_filtered > 0) & (points_3D[:, :, 2] < 6500) 
    
    out_points = points_3D[mask]
    out_colors = imgL_color[mask]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    step = 10
    X = out_points[::step, 0]
    Y = out_points[::step, 1]
    Z = out_points[::step, 2]
    C = out_colors[::step] / 255.0 
    
    # Правильна розстановка осей для 3D-відображення
    ax.scatter(X, Z, -Y, c=C, s=2, marker='.') 
    
    # ЗАБОРОНЯЄМО РОЗТЯГУВАТИ МОДЕЛЬ (фіксуємо пропорції 1:1:1)
    max_range = np.array([X.max()-X.min(), Z.max()-Z.min(), Y.max()-Y.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    mid_y = (-Y.max()-Y.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)
    
    ax.set_xlabel('X (Ширина)')
    ax.set_ylabel('Z (Глибина)')
    ax.set_zlabel('Y (Висота)')
    plt.title('Хмара точок 3D (Правильні пропорції)')
    
    # Нормальний кут огляду
    ax.view_init(elev=5, azim=-80)
    
    plt.show()

if __name__ == "__main__":
    main()