import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(LAB_DIR, "data")
RESULTS_DIR = os.path.join(LAB_DIR, "results")
FIGURES_DIR = os.path.join(LAB_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

img_path = os.path.join(DATA_DIR, "light_day.jpg")


img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Файл {img_path} не знайдено! Перевір назву та шлях.")

print("--- Крок 1. Базова інформація про зображення ---")
print(f"img.shape: {img_bgr.shape}")
print(f"img.dtype: {img_bgr.dtype}")
print(f"img.min():   {img_bgr.min()}")
print(f"img.max():   {img_bgr.max()}\n")


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_bgr)
axes[0].set_title("Без перетворення (BGR як RGB)")
axes[0].axis("off")

axes[1].imshow(img_rgb)
axes[1].set_title("Коректне перетворення (RGB)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_bgr_vs_rgb.png"), dpi=300)
plt.close()
print("Збережено рисунок: figures/fig1_bgr_vs_rgb.png")


bright_overflow = img_bgr + 50

bright_correct = cv2.add(img_bgr, 50)


overflow_mask = (img_bgr.astype(np.int16) + 50) > 255
num_overflowed_pixels = np.sum(overflow_mask)
print(f"--- Крок 3. Переповнення uint8 ---")
print(f"Кількість пікселів із переповненням: {num_overflowed_pixels}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(bright_overflow, cv2.COLOR_BGR2RGB))
axes[0].set_title("img + 50 (uint8 переповнення)")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(bright_correct, cv2.COLOR_BGR2RGB))
axes[1].set_title("cv2.add(img, 50) (насичення)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_uint8_overflow.png"), dpi=300)
plt.close()
print("Збережено рисунок: figures/fig2_uint8_overflow.png")


def calculate_psnr(original, compressed):
    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)

qualities = [95, 75, 50, 25, 10]
compression_results = []


png_path = os.path.join(RESULTS_DIR, "compressed_lossless.png")
cv2.imwrite(png_path, img_bgr)
png_size_kb = os.path.getsize(png_path) / 1024.0

for q in qualities:
    jpeg_path = os.path.join(RESULTS_DIR, f"compressed_q{q}.jpg")
    cv2.imwrite(jpeg_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    
    file_size_kb = os.path.getsize(jpeg_path) / 1024.0
    img_decompressed = cv2.imread(jpeg_path)
    psnr_val = calculate_psnr(img_bgr, img_decompressed)
    
    compression_results.append({
        "quality": q,
        "size_kb": round(file_size_kb, 2),
        "psnr_db": round(psnr_val, 2)
    })


csv_path = os.path.join(RESULTS_DIR, "compression_metrics.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["quality", "size_kb", "psnr_db"])
    writer.writeheader()
    writer.writerows(compression_results)

print(f"\nЗбережено числову таблицю: results/compression_metrics.csv")


sizes = [res["size_kb"] for res in compression_results]
psnrs = [res["psnr_db"] for res in compression_results]

plt.figure(figsize=(7, 4.5))
plt.plot(sizes, psnrs, marker="o", color="crimson", linewidth=2, label="JPEG")
for res in compression_results:
    plt.annotate(f"Q={res['quality']}", (res["size_kb"], res["psnr_db"]),
                 textcoords="offset points", xytext=(5, 5), ha="left", fontsize=9)

plt.title("Залежність метрики якості PSNR від розміру файлу")
plt.xlabel("Розмір файлу [КБ]")
plt.ylabel("PSNR [дБ]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_psnr_vs_filesize.png"), dpi=300)
plt.close()
print("Збережено графік: figures/fig3_psnr_vs_filesize.png")


print("\n--- Крок 5. Метадані EXIF ---")
try:
    pil_img = Image.open(img_path)
    exif_data = pil_img.getexif()
    exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()} if exif_data else {}
    
    iso = exif_dict.get("ISOSpeedRatings", "Не знайдено (можливо, очищено месенджером)")
    exposure = exif_dict.get("ExposureTime", "Не знайдено")
    focal_length = exif_dict.get("FocalLength", "Не знайдено")
    
    print(f"ISO: {iso}")
    print(f"Витримка (Exposure Time): {exposure} с")
    print(f"Фокусна відстань (Focal Length): {focal_length} мм")
except Exception as e:
    print(f"Помилка читання EXIF: {e}")