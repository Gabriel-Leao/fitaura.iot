import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "model/model.savedmodel"
labels_path = "model/labels.txt"

model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

nutrition_info = {
    "Beans":        {"calories": 347, "protein": 21,  "carbs": 63, "fat": 1.2},
    "Bread":        {"calories": 265, "protein": 9,   "carbs": 49, "fat": 3.2},
    "Chicken":      {"calories": 165, "protein": 31,  "carbs": 0,  "fat": 3.6},
    "French Fries": {"calories": 312, "protein": 3.4, "carbs": 41, "fat": 15},
    "Rice":         {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "Steak":        {"calories": 271, "protein": 25,  "carbs": 0,  "fat": 19},
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = (img / 127.5) - 1.0

    output = model(img)
    prediction = tf.squeeze(output["sequential_3"]).numpy()
    index = np.argmax(prediction)
    label = class_names[index]
    confidence = prediction[index]

    if confidence >= 0.9:
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        label_key = label.strip()
        info = nutrition_info.get(label_key)
        if info:
            text = f"{info['calories']} kcal | P: {info['protein']}g | C: {info['carbs']}g | F: {info['fat']}g"
            cv2.putText(frame, text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Food Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
