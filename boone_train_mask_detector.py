# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize the learning rate, number of epochs to train for and the batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Sets the directory for the training data and sets the categories of data
Dir = "dataset"
Categories = ["with_mask", "without_mask"]

print("[INFO] Loading images...")

# Initialize the data and labels list
data = []
labels = []

for category in Categories:
    path = os.path.join(Dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# Preform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Create numpy arrays out of the lists
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Create training and test datasets
test_size = 0.2
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size,
                                                    stratify=labels, random_state=random_state)

# Construct the training image generator for data augmentation
# It altars the same image to create more training and testing data

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Loads the MobileNetV2 network, ensuring the head FC layer sets are left off
base_model = MobileNetV2(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

# Place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=base_model.input, outputs=head_model)

# Loops over all the layers in the base model and freezes them so they will
# not be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# Compile our model
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Trains the head of the network
print("[INFO] Training head...")
H = model.fit(aug.flow(X_train, y_train, batch_size=BS),
              steps_per_epoch=len(X_train) // BS,
              validation_data=(X_test, y_test),
              validation_steps=len(X_test) // BS,
              verbose=1,
              epochs=EPOCHS)

# Make predictions on the testing set
print("[INFO Evaluating network...")
pred_idxs = model.predict(X_test, batch_size=BS)

# For each image in the testing set we need to find the index of the
# label with the corresponding largest predicted probability
pred_idxs = np.argmax(pred_idxs, axis=1)

# Show a formatted classification report
print(classification_report(y_test.argmax(axis=1), pred_idxs,
                            target_names=lb.classes_))

# Serialize the model to disk
print("[INFO] Saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
