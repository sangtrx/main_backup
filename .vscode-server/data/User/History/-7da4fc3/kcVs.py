import os
import keras
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths to the train, val, and test datasets
train_path = "/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/train"
val_path = "/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/val"
test_path = "/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/test"

# Resize one edge of the images to 256 and pad to keep the aspect ratio
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Create the ResNet50 model
model = ResNet50(include_top=False, weights="imagenet")

# Add a new classification layer
input_shape = model.inputs[0].shape.as_list()
model = keras.Model(input_shape, keras.layers.Dense(2, activation="softmax")(model.outputs))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit_generator(
    datagen.flow(train_path, train_labels, batch_size=32),
    steps_per_epoch=len(train_labels) // 32,
    epochs=100,
    validation_data=datagen.flow(val_path, val_labels, batch_size=32),
    validation_steps=len(val_labels) // 32,
    callbacks=[
        ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True),
        EarlyStopping(patience=10),
    ],
)

# Evaluate the model on the test set
loss, accuracy, f1_score, precision, recall = model.evaluate(
    datagen.flow(test_path, test_labels, batch_size=32), verbose=0
)

print("Loss:", loss)
print("Accuracy:", accuracy)
print("F1 score:", f1_score)
print("Precision:", precision)
print("Recall:", recall)

# Save the plot of the loss and accuracy
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.title("Training")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig("loss_accuracy.png")

# Save the plot of the F1 score
plt.plot(history.history["f1_score"], label="F1 score")
plt.title("F1 score")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig("f1_score.png")