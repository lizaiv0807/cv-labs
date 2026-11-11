import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        dim_line = f.readline().decode('ascii').strip()
        width, height = map(int, dim_line.split())
        scale = float(f.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        data = np.reshape(data, (height, width))
        return np.flipud(data)

def main():
    left_img = cv2.imread('img/im0.png', cv2.IMREAD_COLOR)
    
    imgL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    imgR = cv2.imread('img/im1.png', cv2.IMREAD_GRAYSCALE)
    
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192, blockSize=15, uniquenessRatio=5,
                                 P1=8*1*15**2, P2=32*1*15**2, disp12MaxDiff=1, speckleWindowSize=100, speckleRange=32)
    disp_sgbm = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0
    
    disp_gt = read_pfm('img/disp0.pfm')
    disp_gt[np.isinf(disp_gt)] = 0
    
    valid_mask = (disp_sgbm > 0) & (disp_gt > 0)
    
    abs_error = np.abs(disp_sgbm[valid_mask] - disp_gt[valid_mask])
    
    bad_pixels = np.sum(abs_error > 1.0)
    total_valid = np.sum(valid_mask)
    bad_pixel_ratio = (bad_pixels / total_valid) * 100
    
    print(f"Крок 13. Частка пікселів з похибкою > 1 піксель: {bad_pixel_ratio:.2f}%")
    print(f"Загальне покриття нашої карти: {np.sum(disp_sgbm > 0) / disp_sgbm.size * 100:.2f}%")

    f = 3997.684
    B = 193.001
    
    depth_map = np.zeros_like(disp_sgbm)
    depth_map[disp_sgbm > 0] = (f * B) / disp_sgbm[disp_sgbm > 0]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_map, cmap='viridis', vmin=0, vmax=20000)
    plt.colorbar(label='Глибина (міліметри)')
    plt.title('Карта глибини (Z = f * B / d)')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()