# Extracting-Attributes-from-Fashion-Images
 
A model made for a private Kaggle competition with the aim to accurately identify/extract the trouser fit type from fashion apparel trousers images. Participants will make predictions on a diverse set of fashion images.

There were two different models trained for this purpose. 
 -> One is the conventional Convolutional Neural Network and,
 -> EfficientNet
Both models are trained using the same dataset. I have used ImageDataGenerator for the model to receive new variations of the images at each epoch.

### Note:- ImageDataGenerator is a class of Image preprocessing in Keras. It is used to generate batches of data with real-time data augmentation. rescale is one of its arguments, whereas the others that you mention (like batch_size, and target_size) are not part of its argument list. These are rather listed under the flow_from_directory, which is a function of Image preprocessing. This function uses ImageDataGenerator as the base class. You can find more details regarding the various functions and their corresponding argument lists in the Keras documentation for image preprocessing. (Source:- https://stackoverflow.com/questions/49844416/two-questions-about-imagedatagenerator-class-in-keras)
