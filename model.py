import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the CSV file containing labels for the training images
labels_df = pd.read_csv('train.csv')

train_data_dir = 'D:\Work\Projects02\Fashion\\train\data'
test_data_dir = 'D:\Work\Projects02\Fashion\\test'
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)


# Convert the integer labels to strings
labels_df['label'] = labels_df['label'].astype(str)
batch_size = 64
img_height = 112
img_width = 112

# Load the training data
train_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=train_data_dir,
    x_col='file_name',  # Column containing the image filenames
    y_col='label',  # Column containing the labels
    color_mode = "rgb",
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

# Create a deep learning model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the trained model
model.save('trouser_fit_type_model.h5')

# Create the test data generator
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory = test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode= None,
    shuffle=False
)

# Generate predictions for the test data
test_predictions = model.predict(test_generator)
predicted_labels = test_predictions.argmax(axis=1)

# Load the test image filenames
test_filenames = test_generator.filenames

# Create a DataFrame for the submission
submission_data = {'file_name': test_filenames, 'label': predicted_labels}
submission_df = pd.DataFrame(submission_data)

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")