import cv2
import time
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    img_path = 'data/5_bookshelf.jpg'
    img_color = cv2.imread(img_path)
    
    if img_color is None:
        print(f"Помилка: не вдалося завантажити {img_path}")
    else:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create(nfeatures=1000)
        orb = cv2.ORB_create(nfeatures=1000)
        
        start_time = time.time()
        kp_sift, des_sift = sift.detectAndCompute(img_gray, None)
        sift_time = time.time() - start_time
        
        start_time = time.time()
        kp_orb, des_orb = orb.detectAndCompute(img_gray, None)
        orb_time = time.time() - start_time
        
        print("-" * 40)
        print("SIFT Детектор:")
        print(f"Кількість точок: {len(kp_sift)}")
        if des_sift is not None:
            print(f"Розмірність дескриптора: {des_sift.shape[1]}")
            print(f"Тип даних дескриптора: {des_sift.dtype}")
        print(f"Час обчислення: {sift_time:.4f} секунд")
        
        print("-" * 40)
        print("ORB Детектор:")
        print(f"Кількість точок: {len(kp_orb)}")
        if des_orb is not None:
            print(f"Розмірність дескриптора: {des_orb.shape[1]}")
            print(f"Тип даних дескриптора: {des_orb.dtype}")
        print(f"Час обчислення: {orb_time:.4f} секунд")
        print("-" * 40)
        
        img_sift = cv2.drawKeypoints(img_color, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_orb = cv2.drawKeypoints(img_color, kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imwrite('results/sift_keypoints.jpg', img_sift)
        cv2.imwrite('results/orb_keypoints.jpg', img_orb)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
        ax1.set_title("SIFT")
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
        ax2.set_title("ORB")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('figures/part_c_plot.png', bbox_inches='tight')
        plt.show()