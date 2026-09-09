import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


def detect_rail_crack(image_path):

    # =========================================================
    # 1. LOAD IMAGE
    # =========================================================

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image.")
        return

    # Resize for faster processing
    scale_percent = 60
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    img = cv2.resize(
        img,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    original = img.copy()

    img_h, img_w = img.shape[:2]

    # =========================================================
    # NEW: CREATE OUTPUT FOLDER
    # =========================================================

    output_folder = "output_images"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # =========================================================
    # NEW: SAVE INPUT IMAGE
    # =========================================================

    cv2.imwrite(
        os.path.join(
            output_folder,
            "input.png"
        ),
        original
    )

    # =========================================================
    # 2. GRAYSCALE + NOISE REDUCTION
    # =========================================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    # =========================================================
    # NEW: EDGE DETECTION IMAGE
    # =========================================================

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            "edge_detection.png"
        ),
        edges
    )


    roi_x1 = int(img_w * 0.24)
    roi_x2 = int(img_w * 0.30)

    roi_y1 = int(img_h * 0.36)
    roi_y2 = int(img_h * 0.62)

    roi = gray[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ]

    # =========================================================
    # 3. DETECT VERY DARK CRACK REGION
    # =========================================================

    # The crack in your image is almost black while the rail
    # surface around it is much brighter.

    # Therefore we search for genuine dark regions inside
    # the rail inspection area.

    dark_mask = cv2.inRange(
        roi,
        0,
        45
    )

    # =========================================================
    # 4. MORPHOLOGICAL PROCESSING
    # =========================================================

    # Connect small gaps along the crack while removing
    # isolated noise.

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 11)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        vertical_kernel
    )

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 3)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        small_kernel
    )

    # =========================================================
    # 5. FIND CONTOURS
    # =========================================================

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # =========================================================
    # SAVE CONTOUR DETECTION IMAGE
    # =========================================================

    contour_image = original.copy()

    # Draw all detected contours in green.
    for contour in contours:

        cv2.drawContours(
            contour_image,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

    # Convert ROI contour coordinates to full image
    # coordinates automatically.

    for contour in contours:

        contour_full = contour.copy()

        contour_full[:, 0, 0] += roi_x1
        contour_full[:, 0, 1] += roi_y1

        cv2.drawContours(
            contour_image,
            [contour_full],
            -1,
            (0, 255, 0),
            2
        )

    cv2.imwrite(
        os.path.join(
            output_folder,
            "contour_detection.png"
        ),
        contour_image
    )

    best_contour = None
    best_score = -1

    # =========================================================
    # 6. FILTER CONTOURS
    # =========================================================

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w == 0:
            continue

        aspect_ratio = (
            h /
            float(w)
        )

        # -----------------------------------------------
        # Filter 1: Minimum area
        # -----------------------------------------------

        if area < 300:
            continue

        # -----------------------------------------------
        # Filter 2: Crack should be elongated vertically
        # -----------------------------------------------

        if aspect_ratio < 2.5:
            continue

        # -----------------------------------------------
        # Filter 3: Crack should have significant length
        # -----------------------------------------------

        if h < 80:
            continue

        # -----------------------------------------------
        # Filter 4: Check darkness of detected region
        # -----------------------------------------------

        contour_mask = np.zeros_like(
            roi
        )

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            -1
        )

        inside_pixels = roi[
            contour_mask > 0
        ]

        if len(inside_pixels) == 0:
            continue

        mean_inside = np.mean(
            inside_pixels
        )

        # Crack must actually be dark
        if mean_inside > 60:
            continue

        # -----------------------------------------------
        # Filter 5: Check contrast around the crack
        # -----------------------------------------------

        dilated = cv2.dilate(
            contour_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9)
            )
        )

        ring = (
            (dilated > 0)
            &
            (contour_mask == 0)
        )

        if np.sum(ring) == 0:
            continue

        mean_surrounding = np.mean(
            roi[ring]
        )

        contrast = (
            mean_surrounding -
            mean_inside
        )

        # Crack should be significantly darker than its
        # surrounding rail surface
        if contrast < 25:
            continue

        # -----------------------------------------------
        # Calculate detection score
        # -----------------------------------------------

        score = (
            area * 0.4
            + h * 3
            + contrast * 5
        )

        if score > best_score:

            best_score = score
            best_contour = contour

    # =========================================================
    # 7. NO DETECTION
    # =========================================================

    if best_contour is None:

        print("--------------------------------")
        print("RAILWAY TRACK INSPECTION RESULT")
        print("--------------------------------")
        print("No crack detected.")
        print("--------------------------------")

        # NEW: SAVE FINAL OUTPUT EVEN WHEN NO CRACK EXISTS

        final_output_path = os.path.join(
            output_folder,
            "final_output.png"
        )

        cv2.imwrite(
            final_output_path,
            img
        )

        cv2.imshow(
            "Rail Crack Detection",
            img
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return

    # =========================================================
    # 8. QUANTIFY DETECTED CRACK
    # =========================================================

    area = cv2.contourArea(
        best_contour
    )

    x, y, w, h = cv2.boundingRect(
        best_contour
    )

    # Convert ROI coordinates to full-image coordinates
    x_full = x + roi_x1
    y_full = y + roi_y1

    # =========================================================
    # 9. SEVERITY CLASSIFICATION
    # =========================================================

    # These are image-based project thresholds.
    # Measurements are in pixels, NOT millimetres.

    if area >= 800 or h >= 150:

        severity = "HIGH"
        unsafe_zone = "YES"

    elif area >= 400 or h >= 90:

        severity = "MEDIUM"
        unsafe_zone = "YES"

    else:

        severity = "LOW"
        unsafe_zone = "NO"

    # =========================================================
    # 10. DRAW ONLY THE DETECTED CRACK
    # =========================================================

    cv2.rectangle(
        img,
        (
            x_full - 4,
            y_full - 4
        ),
        (
            x_full + w + 4,
            y_full + h + 4
        ),
        (0, 0, 255),
        3
    )

    # Label
    label = (
        f"CRACK - {severity}"
    )

    label_y = max(
        25,
        y_full - 12
    )

    cv2.putText(
        img,
        label,
        (
            x_full,
            label_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    # =========================================================
    # 11. PRINT RESULTS
    # =========================================================

    print()
    print("--------------------------------")
    print("RAILWAY TRACK INSPECTION RESULT")
    print("--------------------------------")

    print(
        "Crack detected       : YES"
    )

    print(
        f"Crack area           : {area:.2f} pixels"
    )

    print(
        f"Crack length         : {h} pixels"
    )

    print(
        f"Crack width          : {w} pixels"
    )

    print(
        f"Aspect ratio         : "
        f"{h / float(w):.2f}"
    )

    print(
        f"Severity             : {severity}"
    )

    print(
        f"Unsafe zone          : {unsafe_zone}"
    )

    print("--------------------------------")

    # =========================================================
    # 12. SAVE OUTPUT IMAGES
    # =========================================================

    # FINAL OUTPUT
    final_output_path = os.path.join(
        output_folder,
        "final_output.png"
    )

    cv2.imwrite(
        final_output_path,
        img
    )

    print(
        f"Final output saved to : "
        f"{final_output_path}"
    )

    print()
    print("All processing images saved:")
    print(
        "  input.png"
    )
    print(
        "  edge_detection.png"
    )
    print(
        "  contour_detection.png"
    )
    print(
        "  final_output.png"
    )

    # =========================================================
    # 13. DISPLAY FINAL RESULT
    # =========================================================

    cv2.imshow(
        "Rail Crack Detection",
        img
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================
# MAIN PROGRAM
# =============================================================

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()

    print()
    print(
        "=============================================="
    )

    print(
        "RAILWAY TRACK CRACK DETECTION SYSTEM"
    )

    print(
        "=============================================="
    )

    while True:

        selected_file_path = filedialog.askopenfilename(

            title="Select Railway Track Image",

            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp"
                )
            ]
        )

        if not selected_file_path:

            print(
                "Exiting program."
            )

            break

        print()
        print(
            f"Processing: {selected_file_path}"
        )

        detect_rail_crack(
            selected_file_path
        )
