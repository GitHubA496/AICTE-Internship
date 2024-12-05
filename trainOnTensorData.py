import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Load the Flowers dataset
dataset_name = "tf_flowers"
(ds_train, ds_val), ds_info = tfds.load(
    dataset_name,
    split=["train[:80%]", "train[80%:]"],  # 80-20 split for train and validation
    as_supervised=True,  # Return (image, label) pairs
    with_info=True
)

# Function to preprocess images
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess_image).shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess_image).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Load MobileNetV2 without top layers (pre-trained on ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(ds_info.features["label"].num_classes, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS
)

# Save the model
model.save("mobilenetv2_tf_flowers.h5")

# Evaluate the model
loss, accuracy = model.evaluate(ds_val)
print(f"Validation Accuracy: {accuracy:.2f}")
