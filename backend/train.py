import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- settings (lower IMG to 64 and EPOCHS_FT to 6 if training is too slow) ----
IMG = 96
BATCH = 64
EPOCHS_HEAD = 5
EPOCHS_FT = 12
# --------------------------------------------------------------------------------

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "models"
OUT.mkdir(exist_ok=True)

train_ds = keras.utils.image_dataset_from_directory(
    DATA / "train", image_size=(IMG, IMG), color_mode="rgb",
    batch_size=BATCH, label_mode="categorical", shuffle=True, seed=42,
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA / "test", image_size=(IMG, IMG), color_mode="rgb",
    batch_size=BATCH, label_mode="categorical", shuffle=False,
)

class_names = train_ds.class_names
print("Classes:", class_names)
(OUT / "class_names.json").write_text(json.dumps(class_names))

# softened class weights (the 'disgust' class has very few images)
counts = [len(list((DATA / "train" / c).glob("*"))) for c in class_names]
total = sum(counts)
class_weight = {i: (total / (len(counts) * c)) ** 0.5 for i, c in enumerate(counts)}
print("Class weights:", class_weight)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
])

base = keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3)
)
base.trainable = False

inputs = keras.Input((IMG, IMG, 3))
x = augment(inputs)
x = base(x, training=False)          # EfficientNet rescales 0-255 inputs internally
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)


def fit(epochs, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              class_weight=class_weight, callbacks=cbs)


print("\n=== Phase 1: training the new head ===")
fit(EPOCHS_HEAD, 1e-3)

print("\n=== Phase 2: fine-tuning top EfficientNet layers ===")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
fit(EPOCHS_FT, 1e-4)

loss, acc = model.evaluate(val_ds)
print(f"\nFinal validation accuracy: {acc:.3f}")
model.save(OUT / "emotion_model.keras")
print("Saved ->", OUT / "emotion_model.keras")