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


texture_path = os.path.join(DATA_DIR, "texture.jpg")

if os.path.exists(texture_path):
    img_tex = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
else:
    x = np.linspace(-20, 20, 1000)
    xx, yy = np.meshgrid(x, x)
    pattern = np.sin(xx**2 + yy**2)
    img_tex = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)

h, w = img_tex.shape[:2]
target_size = (w // 4, h // 4)

down_nearest = cv2.resize(img_tex, target_size, interpolation=cv2.INTER_NEAREST)

sigma = 4 / 2.0  
blurred = cv2.GaussianBlur(img_tex, (9, 9), sigmaX=sigma)
down_gaussian = cv2.resize(blurred, target_size, interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(down_nearest, cmap="gray")
axes[0].set_title("Зменшення в 4 рази: INTER_NEAREST (Муар)")
axes[0].axis("off")

axes[1].imshow(down_gaussian, cmap="gray")
axes[1].set_title("Зменшення в 4 рази: GaussianBlur + Area")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_aliasing_moire.png"), dpi=300)
plt.close()
print("Збережено рисунок: figures/fig5_aliasing_moire.png")

base_img_path = os.path.join(DATA_DIR, "light_day.jpg")
base_img = cv2.imread(base_img_path)
if base_img is None:
    raise FileNotFoundError(f"Не знайдено {base_img_path}")


cy, cx = base_img.shape[0] // 2, base_img.shape[1] // 2
hr_crop = base_img[cy:cy + 200, cx:cx + 200]


lr_crop = cv2.resize(hr_crop, (25, 25), interpolation=cv2.INTER_AREA)

methods = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4
}

def calculate_psnr(orig, recon):
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)

interp_results = []
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, (name, flag) in enumerate(methods.items()):
    upscaled = cv2.resize(lr_crop, (200, 200), interpolation=flag)
    psnr_score = calculate_psnr(hr_crop, upscaled)
    interp_results.append({"method": name, "psnr_db": round(psnr_score, 2)})
    
    axes[idx].imshow(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f"{name}\nPSNR: {psnr_score:.2f} dB")
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig6_interpolation_comparison.png"), dpi=300)
plt.close()
print("Збережено рисунок: figures/fig6_interpolation_comparison.png")


csv_interp_path = os.path.join(RESULTS_DIR, "interpolation_metrics.csv")
with open(csv_interp_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "psnr_db"])
    writer.writeheader()
    writer.writerows(interp_results)
print("Збережено таблицю: results/interpolation_metrics.csv")


gammas = [0.5, 1.0, 2.2]
fig_gamma, axes_gamma = plt.subplots(1, 3, figsize=(15, 5))

for i, g in enumerate(gammas):
    # Побудова Look-Up Table (LUT) для нормалізованих значень [0, 1] -> [0, 255]
    lut = np.array([((val / 255.0) ** (1.0 / g)) * 255 for val in range(256)]).astype("uint8")
    corrected = cv2.LUT(base_img, lut)
    
    axes_gamma[i].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes_gamma[i].set_title(f"Gamma = {g}")
    axes_gamma[i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig7_gamma_images.png"), dpi=300)
plt.close()


x_vals = np.linspace(0, 1, 256)
plt.figure(figsize=(6, 5))
for g in gammas:
    y_vals = x_vals ** (1.0 / g)
    plt.plot(x_vals, y_vals, label=f"$\\gamma = {g}$", linewidth=2)

plt.title("Передавальні характеристики гамма-перетворення")
plt.xlabel("Вхідна яскравість (нормалізована)")
plt.ylabel("Вихідна яскравість (нормалізована)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig8_gamma_curves.png"), dpi=300)
plt.close()
print("Збережено рисунки: figures/fig7_gamma_images.png та figures/fig8_gamma_curves.png")


I1, I2 = 0.2, 0.8
gamma_val = 2.2
linear_avg = (I1 + I2) / 2.0
gamma_avg = ((I1 ** (1.0 / gamma_val) + I2 ** (1.0 / gamma_val)) / 2.0) ** gamma_val
print("\n--- Крок 14. Нелінійність гамма-кодування ---")
print(f"Фізичне середнє освітленостей (Linear avg): {linear_avg:.4f}")
print(f"Середнє в гамма-просторі з декодуванням:    {gamma_avg:.4f}")