import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Load the CSV file containing labels for the training images
labels_df = pd.read_csv('train.csv')

train_data_dir = 'D:\Work\Projects02\Fashion\\train\data'
test_data_dir = 'D:\Work\Projects02\Fashion\\test'

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Convert the integer labels to strings
labels_df['label'] = labels_df['label'].astype(str)
batch_size = 32
img_height = 224
img_width = 224

# Load the training data
train_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=train_data_dir,
    x_col='file_name',
    y_col='label',
    color_mode="rgb",
    subset='training',
    batch_size=batch_size,
    target_size=(img_height, img_width),
    class_mode='categorical'
)

# Load the validation data
validation_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=train_data_dir,
    x_col='file_name',
    y_col='label',
    subset='validation',
    batch_size=batch_size,
    target_size=(img_height, img_width),
    class_mode='categorical'
)

# Use EfficientNetB0 as a base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Build a custom classifier on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Define callbacks for training
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False),
    ReduceLROnPlateau(factor=0.1, patience=3),
    EarlyStopping(patience=10, restore_best_weights=True)
]

# Train the model
epochs = 1
# Train the model
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Convert the EagerTensor objects to numpy arrays before saving the history
history.history['loss'] = [float(loss) for loss in history.history['loss']]
history.history['accuracy'] = [float(acc) for acc in history.history['accuracy']]
history.history['val_loss'] = [float(loss) for loss in history.history['val_loss']]
history.history['val_accuracy'] = [float(acc) for acc in history.history['val_accuracy']]

# Save the trained model
model.save('trouser_fit_type_model', save_format='tf')

# Load the trained model
model.load_weights('best_model.h5')

# Create the test data generator
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode=None,
    shuffle=False
)


# Load the saved model
loaded_model = tf.keras.models.load_model('trouser_fit_type_model')

# Use the loaded model for inference
test_predictions = loaded_model.predict(test_generator)
predicted_labels = test_predictions.argmax(axis=1)

# Load the test image filenames
test_filenames = [os.path.basename(file) for file in test_generator.filenames]

# Create a DataFrame for the submission
submission_data = {'file_name': test_filenames, 'label': predicted_labels}
submission_df = pd.DataFrame(submission_data)

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
