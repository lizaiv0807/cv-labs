import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(LAB_DIR, "data")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
FIGURES_DIR = os.path.join(LAB_DIR, "figures")


img_day_bgr = cv2.imread(os.path.join(DATA_DIR, "light_day.jpg"))
img_lamp_bgr = cv2.imread(os.path.join(DATA_DIR, "light_lamp.jpg"))
mask_gt = cv2.imread(os.path.join(DATA_DIR, "mask_day.png"), cv2.IMREAD_GRAYSCALE)

if img_day_bgr is None or img_lamp_bgr is None or mask_gt is None:
    raise FileNotFoundError("Перевірте наявність light_day.jpg, light_lamp.jpg та mask_day.png у lab01/data/")


gt_binary = (mask_gt > 127).astype(np.uint8)

def compute_metrics(pred_mask, gt_mask):
    """Обчислення Accuracy та IoU"""
    pred_bin = (pred_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_mask).sum()
    union = np.logical_or(pred_bin, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    accuracy = (pred_bin == gt_mask).mean()
    return round(float(accuracy), 4), round(float(iou), 4)


img_day_rgb = cv2.cvtColor(img_day_bgr, cv2.COLOR_BGR2RGB)
img_lamp_rgb = cv2.cvtColor(img_lamp_bgr, cv2.COLOR_BGR2RGB)


lower_rgb = np.array([0, 100, 110])
upper_rgb = np.array([60, 200, 210])

mask_rgb_day = cv2.inRange(img_day_rgb, lower_rgb, upper_rgb)
mask_rgb_lamp = cv2.inRange(img_lamp_rgb, lower_rgb, upper_rgb)

acc_rgb_day, iou_rgb_day = compute_metrics(mask_rgb_day, gt_binary)
acc_rgb_lamp, iou_rgb_lamp = compute_metrics(mask_rgb_lamp, gt_binary)


img_day_hsv = cv2.cvtColor(img_day_bgr, cv2.COLOR_BGR2HSV)
img_lamp_hsv = cv2.cvtColor(img_lamp_bgr, cv2.COLOR_BGR2HSV)

lower_hsv = np.array([80, 80, 70])
upper_hsv = np.array([105, 255, 255])

mask_hsv_day = cv2.inRange(img_day_hsv, lower_hsv, upper_hsv)
mask_hsv_lamp = cv2.inRange(img_lamp_hsv, lower_hsv, upper_hsv)

acc_hsv_day, iou_hsv_day = compute_metrics(mask_hsv_day, gt_binary)
acc_hsv_lamp, iou_hsv_lamp = compute_metrics(mask_hsv_lamp, gt_binary)


metrics_table = [
    {"space": "RGB", "day_accuracy": acc_rgb_day, "day_iou": iou_rgb_day,
     "lamp_accuracy": acc_rgb_lamp, "lamp_iou": iou_rgb_lamp},
    {"space": "HSV", "day_accuracy": acc_hsv_day, "day_iou": iou_hsv_day,
     "lamp_accuracy": acc_hsv_lamp, "lamp_iou": iou_hsv_lamp}
]

csv_path = os.path.join(RESULTS_DIR, "color_segmentation_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["space", "day_accuracy", "day_iou", "lamp_accuracy", "lamp_iou"])
    writer.writeheader()
    writer.writerows(metrics_table)

print("--- Крок 11. Результати сегментації ---")
for row in metrics_table:
    print(f"Простір {row['space']}:")
    print(f"  Денне світло:  Accuracy = {row['day_accuracy']}, IoU = {row['day_iou']}")
    print(f"  Лампа (вечір): Accuracy = {row['lamp_accuracy']}, IoU = {row['lamp_iou']}")


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(mask_rgb_day, cmap="gray")
axes[0, 0].set_title(f"RGB - Денне (IoU: {iou_rgb_day})")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_rgb_lamp, cmap="gray")
axes[0, 1].set_title(f"RGB - Лампа (IoU: {iou_rgb_lamp})")
axes[0, 1].axis("off")

axes[1, 0].imshow(mask_hsv_day, cmap="gray")
axes[1, 0].set_title(f"HSV - Денне (IoU: {iou_hsv_day})")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_hsv_lamp, cmap="gray")
axes[1, 1].set_title(f"HSV - Лампа (IoU: {iou_hsv_lamp})")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_color_segmentation.png"), dpi=300)
plt.close()
print("Збережено рисунок: figures/fig4_color_segmentation.png")