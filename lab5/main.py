import cv2
import numpy as np
import time
import os
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
    left_img_path = 'img/im0.png'
    right_img_path = 'img/im1.png'
    
    imgL = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    if imgL is None or imgR is None:
        print("Помилка завантаження зображень.")
        return

    # Створюємо папку для збереження картинок експериментів
    os.makedirs('results', exist_ok=True)

    # Параметри для сітки експериментів з методички
    num_disparities_list = [64, 128, 192]
    block_size_list = [5, 9, 15, 21]
    uniqueness_ratio_list = [5, 10, 15]

    # Заголовок таблиці для консолі
    print(f"| {'Алгоритм':<8} | {'numDisp':<7} | {'blockSz':<7} | {'uniqRat':<7} | {'Час (с)':<7} | {'Покриття (%)':<12} |")
    print("|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*14 + "|")

    for nd in num_disparities_list:
        for bs in block_size_list:
            for ur in uniqueness_ratio_list:
                
                # --- Експеримент для BM ---
                bm = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
                bm.setUniquenessRatio(ur)
                
                start_time = time.time()
                disp_bm = bm.compute(imgL, imgR).astype(np.float32) / 16.0
                time_bm = time.time() - start_time
                
                # Покриття: рахуємо пікселі, де диспаратність >= 0 (невідомі пікселі мають значення -1)
                cov_bm = np.sum(disp_bm >= 0) / disp_bm.size * 100
                
                print(f"| {'BM':<8} | {nd:<7} | {bs:<7} | {ur:<7} | {time_bm:<7.3f} | {cov_bm:<12.2f} |")
                
                # Зберігаємо картинку, щоб ти могла дати "візуальну оцінку"
                plt.imsave(f'results/BM_nd{nd}_bs{bs}_ur{ur}.png', disp_bm, cmap='jet', vmin=0, vmax=nd)

                # --- Експеримент для SGBM ---
                sgbm = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=nd,
                    blockSize=bs,
                    uniquenessRatio=ur,
                    P1=8 * 1 * bs**2,
                    P2=32 * 1 * bs**2,
                    disp12MaxDiff=1,
                    speckleWindowSize=100,
                    speckleRange=32
                )
                
                start_time = time.time()
                disp_sgbm = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0
                time_sgbm = time.time() - start_time
                
                cov_sgbm = np.sum(disp_sgbm >= 0) / disp_sgbm.size * 100
                
                print(f"| {'SGBM':<8} | {nd:<7} | {bs:<7} | {ur:<7} | {time_sgbm:<7.3f} | {cov_sgbm:<12.2f} |")
                
                plt.imsave(f'results/SGBM_nd{nd}_bs{bs}_ur{ur}.png', disp_sgbm, cmap='jet', vmin=0, vmax=nd)

    print("\nЕксперименти завершено! Усі картинки збережено у папку 'results'.")

if __name__ == "__main__":
    main()