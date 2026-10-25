import numpy as np
import matplotlib.pyplot as plt

# 10 клітинок по горизонталі, 7 по вертикалі -> 9x6 внутрішніх кутів
rows, cols = 7, 10
board = np.indices((rows, cols)).sum(axis=0) % 2

# Розміри стандартного аркуша A4 (альбомна орієнтація: 297 x 210 мм у дюймах)
fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300)

# Малюємо шахову дошку
ax.imshow(board, cmap="gray", interpolation="nearest")
ax.axis("off")

# Залишаємо широкі білі поля навколо дошки (критично для OpenCV)
plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)

# Зберігаємо файл для друку
output_file = "chessboard_9x6.png"
plt.savefig(output_file, dpi=300, facecolor="white")
plt.close()

print(f"Шаблон успішно збережено: {output_file}")