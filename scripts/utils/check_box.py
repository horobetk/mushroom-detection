import cv2
import os

IMAGE_PATH = "C:/mushroom_data/test_output/images/train/374847299_685196441.jpg"
LABEL_PATH = "C:/mushroom_data/test_output/labels/train/374847299_685196441.txt"
OUTPUT_PATH = "C:/mushroom_data/check_result.jpg"


def draw_yolo_boxes():
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(LABEL_PATH):
        print(f"[ERR] Missing image or label path")
        return

    img = cv2.imread(IMAGE_PATH)
    height, width, _ = img.shape

    with open(LABEL_PATH, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                cx, cy, w, h = map(float, parts[1:5])

                x_center, y_center = int(cx * width), int(cy * height)
                box_width, box_height = int(w * width), int(h * height)

                x_min = int(x_center - box_width / 2)
                y_min = int(y_center - box_height / 2)
                x_max = int(x_center + box_width / 2)
                y_max = int(y_center + box_height / 2)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"Class {class_id}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"[OK] Wrote overlay to {OUTPUT_PATH}")


if __name__ == "__main__":
    draw_yolo_boxes()
