import cv2
import numpy as np

IMAGE_PATH = r"input.jpeg"
DRAW_DELAY_MS = 1
LINE_THICKNESS = 2
POINT_SKIP = 2
CANNY_LOW = 80
CANNY_HIGH = 180
COLOR_CLUSTERS = 8
REVEAL_STEPS = 60
FADE_ZONE = 40

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image not found")

h, w = img.shape[:2]

data = img.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

_, labels, centers = cv2.kmeans(
    data, COLOR_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
color_img = centers[labels.flatten()].reshape(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

contours, _ = cv2.findContours(
    edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
)

sketch_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
reveal_mask = np.zeros((h, w), dtype=np.uint8)

cv2.namedWindow("Sketch Reveal", cv2.WINDOW_NORMAL)

for contour in contours:
    if len(contour) < 2:
        continue

    for i in range(POINT_SKIP, len(contour), POINT_SKIP):
        p1 = contour[i - POINT_SKIP][0]
        p2 = contour[i][0]

        cv2.line(
            sketch_canvas,
            tuple(p1),
            tuple(p2),
            (0, 0, 0),
            LINE_THICKNESS
        )

        cv2.line(
            reveal_mask,
            tuple(p1),
            tuple(p2),
            255,
            LINE_THICKNESS + 2
        )

        mask_3ch = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR)
        revealed_color = cv2.bitwise_and(color_img, mask_3ch)

        final_frame = cv2.addWeighted(
            sketch_canvas, 1.0,
            revealed_color, 1.0,
            0
        )
    

        cv2.imshow("Sketch Reveal", final_frame)

        if cv2.waitKey(DRAW_DELAY_MS) == 25:
            break

cv2.imshow("Sketch Reveal", final_frame)
cv2.waitKey(800)

for step in range(REVEAL_STEPS + 1):
    mask = np.zeros((h, w), dtype=np.float32)
    reveal_height = int((step / REVEAL_STEPS) * h)

    if reveal_height > 0:
        mask[h - reveal_height : h, :] = 1.0
        y_start = max(h - reveal_height - FADE_ZONE, 0)
        for y in range(y_start, h - reveal_height):
            mask[y, :] = (y - y_start) / FADE_ZONE

    mask_3ch = cv2.merge([mask, mask, mask])

    blended = (
        final_frame * (1 - mask_3ch) +
        img * mask_3ch
    ).astype(np.uint8)

    cv2.imshow("Sketch Reveal", blended)
    cv2.waitKey(30)

cv2.waitKey(0)
cv2.destroyAllWindows()
