import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

#load dataset
(x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()

#reshape for cnn
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

#class names
class_name = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Build Cnn model
model = keras.Sequential([
keras.Input(shape=(28,28,1)),
layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64, (3,3), activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.3),
layers.Dense(10, activation='softmax')
])

#compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train model
model_train=model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_test,y_test))

#Evaluate Model
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

#graph
plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(model_train.history['accuracy'], label="Train")
plt.plot(model_train.history['val_accuracy'], label="Validation")
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(model_train.history['loss'], label="Train")
plt.plot(model_train.history['val_loss'], label="Validation")
plt.title("Loss")
plt.legend()

plt.show()


#predict
predictions = model.predict(x_test)

index = 0
plt.imshow(x_test[index].reshape(28,28), cmap="gray")
plt.title(f"Pred: {class_name[np.argmax(predictions[index])]} | Actual: {class_name[y_test[index]]}")
plt.show()
