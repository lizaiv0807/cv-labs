"""Camera calibration from chessboard photographs (OpenCV).
Run from lab2: python src/calibrate.py
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
DATA_DIR = LAB_DIR / 'data'
FIGURES_DIR = LAB_DIR / 'figures'
RESULTS_DIR = LAB_DIR / 'results'
SQUARE_SIZE_MM = 25.0  # Verify this is the actual printed square size.
MIN_IMAGES = 8
OUTLIER_THRESHOLD_PX = 2.5
# Number of INTERNAL corners, columns x rows. Choose a verified physical grid.
PATTERNS = [(9, 7), (7, 9), (8, 6), (6, 8), (9, 6), (6, 9),
            (8, 5), (5, 8), (7, 6), (6, 7), (7, 5), (5, 7),
            (6, 5), (5, 6), (5, 4), (4, 5)]
# Set to (cols, rows) after visually verifying the grid on every image.
FORCE_PATTERN = None


def find_corners(gray, pattern):
    flags = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY |
             cv2.CALIB_CB_NORMALIZE_IMAGE)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if not ok or corners is None or len(corners) != pattern[0] * pattern[1]:
        return None
    return corners.astype(np.float32)


def object_grid(pattern):
    cols, rows = pattern
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= SQUARE_SIZE_MM
    return points


def fit(objpoints, imgpoints, image_size):
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)
    return cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None,
                               criteria=criteria)


def frame_errors(objpoints, imgpoints, K, dist, rvecs, tvecs):
    errors = []
    for obj, corners, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))
    return np.array(errors)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f'Missing image folder: {DATA_DIR}')

    paths = sorted(p for p in DATA_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise RuntimeError(f'No images found in {DATA_DIR}')
    print(f'Images found: {len(paths)}')
    images = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f'[SKIP] Cannot read {path.name}')
            continue
        images.append((path, img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    if not images:
        raise RuntimeError('No readable images')

    patterns = [FORCE_PATTERN] if FORCE_PATTERN else PATTERNS
    counts = []
    print('\nChecking internal-corner patterns:')
    for pattern in patterns:
        count = sum(find_corners(gray, pattern) is not None for _, _, gray in images)
        counts.append((count, pattern[0] * pattern[1], pattern))
        print(f'  {pattern}: {count}/{len(images)} images')
    # Prioritize number of usable images, then grid size.
    count, _, pattern = max(counts)
    if count < MIN_IMAGES:
        raise RuntimeError('Too few detected boards. Check the physical grid and photos.')
    print(f'\nSelected pattern: {pattern}; detections: {count}')
    print('IMPORTANT: Check the saved corner overlays. The SAME physical grid must be')
    print('detected in every image. Automatic pattern selection is not proof of validity.')

    objtemplate = object_grid(pattern)
    objpoints, imgpoints, names = [], [], []
    image_size = None
    for path, img, gray in images:
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            print(f'[SKIP] Different image size: {path.name}')
            continue
        corners = find_corners(gray, pattern)
        if corners is None:
            print(f'[SKIP] Board not detected: {path.name}')
            continue
        objpoints.append(objtemplate.copy())
        imgpoints.append(corners)
        names.append(path.name)
        overlay = img.copy()
        cv2.drawChessboardCorners(overlay, pattern, corners, True)
        cv2.imwrite(str(FIGURES_DIR / f'corners_{path.stem}.jpg'), overlay)
        print(f'[OK] {path.name}')
    if len(names) < MIN_IMAGES:
        raise RuntimeError('Too few usable images with matching resolution')

    rms0, K, dist, rvecs, tvecs = fit(objpoints, imgpoints, image_size)
    errors = frame_errors(objpoints, imgpoints, K, dist, rvecs, tvecs)
    threshold = max(OUTLIER_THRESHOLD_PX, float(np.median(errors)) * 2)
    good = [i for i, err in enumerate(errors) if err <= threshold]
    rejected = []
    if MIN_IMAGES <= len(good) < len(names):
        rejected = [name for i, name in enumerate(names) if i not in good]
        objpoints = [objpoints[i] for i in good]
        imgpoints = [imgpoints[i] for i in good]
        names = [names[i] for i in good]
        rms, K, dist, rvecs, tvecs = fit(objpoints, imgpoints, image_size)
    else:
        rms = rms0
    errors = frame_errors(objpoints, imgpoints, K, dist, rvecs, tvecs)

    report = [f'Internal corners: {pattern}', f'Square size: {SQUARE_SIZE_MM} mm',
              f'Image size: {image_size}', f'Images found: {len(paths)}',
              f'Images used: {len(names)}', f'Initial RMS: {rms0:.6f} px',
              f'Final RMS: {rms:.6f} px', '\nCamera matrix:', str(K),
              '\nDistortion coefficients:', str(dist), '\nPer-image RMS:']
    report += [f'{name}: {err:.6f} px' for name, err in zip(names, errors)]
    if rejected:
        report += ['\nRejected images:'] + rejected
    (RESULTS_DIR / 'calibration_results.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    np.savez(RESULTS_DIR / 'calibration.npz', camera_matrix=K, dist_coeffs=dist,
             rms=rms, pattern=np.array(pattern), image_size=np.array(image_size))
    plt.figure(figsize=(max(11, len(names) * .48), 6))
    plt.bar(range(len(names)), errors)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.ylabel('Reprojection RMS (pixels)')
    plt.title('Per-image reprojection error')
    plt.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'reprojection_errors.png', dpi=160)
    plt.close()
    print('\n' + '\n'.join(report[:7]))
    print('\nCamera matrix:\n', K)
    print('\nDistortion:\n', dist)
    print('\nSaved results to:', RESULTS_DIR)
    print('Saved overlays and chart to:', FIGURES_DIR)


if __name__ == '__main__':
    main()
