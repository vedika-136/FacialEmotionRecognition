# FacialEmotionRecognition
Facial Emotion Recognition for 5 different human emotions


• Project Overview
  This project involves building and deploying a real-time emotion recognition system using a ResNet50 deep learning model. The system is trained to classify facial expressions into five categories: Anger, Fear, Happy, Neutral, and Sad. The final model is capable of making predictions on both static images and real-time video feed from a webcam.
  
• Project Structure

• Data Preparation:

The dataset is organized with separate folders for each emotion, and the number of images per emotion is counted and displayed.
The dataset is split into training and validation sets using train_test_split from sklearn.

• Data Augmentation and Transformation:
Training images are augmented using transformations like random resizing, horizontal flip, rotation, and color jittering to improve the model's generalization ability.
Validation images are resized and normalized.

• Custom Dataset Class:
A custom MoodDataset class is created, inheriting from PyTorch's Dataset class, to load images and their corresponding labels.

• Model Setup:
A pre-trained ResNet50 model is used, and the final fully connected layer is modified to match the number of emotion classes.
The model is trained using a cross-entropy loss function and the Adam optimizer.
A learning rate scheduler (ReduceLROnPlateau) is used to reduce the learning rate if the validation loss plateaus.

• Training Loop:
The model is trained over multiple epochs, with the loss and accuracy tracked for both training and validation datasets.
The model with the best validation accuracy is saved.

• Model Evaluation:
The trained model is evaluated on test images, and the predictions are displayed alongside the input images.

• Real-time Emotion Detection:
A real-time emotion detection system is implemented using OpenCV to capture video from the webcam.
Detected faces in each frame are processed and fed into the model for emotion prediction, and the result is displayed in real-time.

• Train the Model:
Run the script to start the training process.
The model will be trained and saved as mood_recognition_model1.pth.

• Evaluate the Model:
Test the trained model on static images by placing them in the images list and running the evaluation section of the script.

• Real-time Emotion Detection:
Ensure your webcam is working.
Run the real-time emotion detection section of the script.
The system will display a live feed with detected faces and their predicted emotions.

• Exit Real-time Detection:
Press 'q' to exit the real-time detection loop.

Customization
• Emotion Classes: Modify the emotion_classes list if you have different emotion categories.
• Model Architecture: You can change the pre-trained model to another architecture (e.g., ResNet18, VGG16) based on your needs.
• Training Parameters: Adjust learning rate, batch size, and number of epochs in the training loop to fine-tune performance.

• Results
The final model achieves a validation accuracy of around 83.93% after 50 epochs of training. The loss and accuracy plots are also provided to analyze the training process.

• Conclusion
This project demonstrates a complete pipeline for training and deploying a deep learning model for real-time emotion recognition. The model is capable of processing both static images and real-time video, making it suitable for applications in human-computer interaction, surveillance, and other areas where emotion detection is valuable.
