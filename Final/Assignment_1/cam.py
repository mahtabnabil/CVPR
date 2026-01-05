import cv2
import numpy as np
import tensorflow as tf

# Load model
MODEL_PATH = "mnist_dense_model.keras"
THRESHOLD  = 0.70

model = tf.keras.models.load_model(MODEL_PATH)
print("\nModel loaded:", MODEL_PATH)

# Preprocess ROI to MNIST format
def preprocess_roi(roi_bgr):
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # convert to MNIST-like foreground
    _, th = cv2.threshold(
        roi_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # contour detection
    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        blank = np.zeros((28, 28), dtype=np.uint8)
        return np.zeros((1, 784), np.float32), blank, False

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 200:
        blank = np.zeros((28, 28), dtype=np.uint8)
        return np.zeros((1, 784), np.float32), blank, False

    x, y, w, h = cv2.boundingRect(c)

    # padding
    pad = 20
    x0 = max(x-pad, 0)
    y0 = max(y-pad, 0)
    x1 = min(x+w+pad, th.shape[1])
    y1 = min(y+h+pad, th.shape[0])

    digit = th[y0:y1, x0:x1]
    H, W = digit.shape

    # keep aspect ratio into MNIST layout
    if H > W:
        new_h = 20
        new_w = max(1, int(W * (20 / H)))
    else:
        new_w = 20
        new_h = max(1, int(H * (20 / W)))

    digit_resized = cv2.resize(
        digit, (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized

    x_inp = (canvas.astype("float32") / 255.0).reshape(1, 784)
    return x_inp, canvas, True

# Camera Loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

print("Press 'q' to quit.\n")

box_size = 160  # ROI size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]

    x_inp, roi_28, ok = preprocess_roi(roi)

    if ok:
        probs = model.predict(x_inp, verbose=0)[0]
        pred  = int(np.argmax(probs))
        conf  = float(np.max(probs))
    else:
        pred, conf = None, 0.0

    if conf >= THRESHOLD:
        text  = f"Digit: {pred}  Conf: {conf:.2f}"
        color = (0,255,0)
    else:
        text  = f"Uncertain (max={conf:.2f})"
        color = (0,0,255)

    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, text, (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)

    # show model input preview
    preview = cv2.resize(roi_28, (140,140), cv2.INTER_NEAREST)
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    frame[10:150, w-150:w-10] = preview
    cv2.putText(frame, "Model Input", (w-150, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 1)

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
