import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Split into training (80%) and testing (20%) sets randomly
x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# Predict classes for test set
y_pred = np.argmax(model.predict(x_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')


plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, [str(i) for i in range(10)])
plt.yticks(tick_marks, [str(i) for i in range(10)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="black")
plt.show()
random_index = np.random.randint(4, len(x_test))
random_image = x_test[random_index]
true_label = np.argmax(y_test[random_index])

# Plot the random image
plt.figure()
plt.imshow(random_image, cmap='gray')
plt.title(f'True label: {true_label}')
plt.show()
from PIL import Image
def predict_image_from_file(filepath):
    try:
        # Load the image
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to model's input dimensions
        img_array = np.array(img) / 255.0  # Normalize pixel values
      
        # Ensure the image array has the correct shape
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = img_array.reshape(1, 28, 28)  
        # Print the shape of the image array
        print("Image array shape:", img_array.shape)

        # Predict the label
        prediction = model.predict(img_array)
        print(prediction)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Plot the image with the predicted label
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted label: {predicted_label}')
        plt.show()

        return predicted_label

    except Exception as e:
        print(f"Error loading image: {e}")
        return None
# Example usage: Predict an image from file
predict_image_from_file(r'B:\IMLprojfinal\2.png')