#Load + Plot

import tensorflow as tf
import matplotlib.pyplot as plt
''''
# Load the model
model = tf.saved_model.load('/Users/thomaskovarovics/Python/CNN_allImg-1/')

# Plot the learning curves
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()
'''
# crying bc no history

import matplotlib.pyplot as plt

# Define the data for the plot
loss = [3.7205, 0.1729, 0.0837, 0.0772, 0.0732, 0.0914, 0.1002, 0.0903, 0.1050, 0.0984, 0.0983, 0.1025, 0.0756, 0.0922, 0.0969, 0.0637, 0.1400, 0.0882, 0.0966, 0.1118]
accuracy = [0.8365, 0.9583, 0.9819, 0.9870, 0.9907, 0.9909, 0.9922, 0.9936, 0.9937, 0.9945, 0.9950, 0.9944, 0.9967, 0.9959, 0.9966, 0.9973, 0.9957, 0.9972, 0.9970, 0.9974]
val_loss = [0.4152, 0.6591, 1.0232, 1.5176, 2.4467, 3.4823, 3.9629, 3.9613, 4.9441, 5.5746, 6.2923, 6.7994, 7.2802, 8.5941, 8.9847, 10.6031, 10.3157, 10.6521, 11.2973, 11.7939]
val_accuracy = [0.9005, 0.9495, 0.9610, 0.9690, 0.9420, 0.9650, 0.9640, 0.9610, 0.9655, 0.9635, 0.9325, 0.9605, 0.9675, 0.9440, 0.9530, 0.9600, 0.9470, 0.9660, 0.9675, 0.9550]

'''# Create the plot
plt.plot(loss, label='Training loss')
plt.plot(accuracy, label='Training accuracy')
plt.plot(val_loss, label='Validation loss')
plt.plot(val_accuracy, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()'''

# Create the first plot
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Create the second plot
plt.plot(accuracy, label='Training accuracy')
plt.plot(val_accuracy, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()