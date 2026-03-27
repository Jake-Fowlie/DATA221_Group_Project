
#Importing Library dependencies 
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import(
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import(
    classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
    )
import os


SEED = 42 # This allows for reproducibility so everytime we run the script, we get the exact same model, weight initializations and data shuffling.
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Load and Data Preprocessing 

TRAIN_DIR = './data/Training' #Storing the data paths in variables for easy recall
TEST_DIR  = './data/Testing'

img_size = (128, 128) # This sets all Images to be resized to 128x128 pixels.

batch_size = 32 # The numbe of images processed together before the model updates it weights

num_classes = 4 #this indicate the number of outputs

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15, 
    horizontal_flip = True, #mirrors horizontally 
    validation_split = 0.20 # 20% validation set
)

#creating a seperate pipeline for validation and test images.
val_test_datagen = ImageDataGenerator(rescale=1./255)


#Loading the training images in batch sizes

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    subset='training',
    seed=SEED
)

#This line of code is almost same as the one above except it takes 20% training set as validation set
#Dont get confused

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = 'rgb',
    subset = 'validation',
    seed = SEED,
)

# Now lets load the test set using the same approach above

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle = False #keeps  the order so predictions match true labels
)

print('Class indices:', train_gen.class_indices)

#Checking for class distribution

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

print('Training class counts:')

for c in classes: 
    count = len(os.listdir(os.path.join(TRAIN_DIR, c))) # looping through each class folder and counting ythe number of image files inside.
    print (f'{c}: {count}')


from sklearn.utils import class_weight

labels = train_gen.classes # array of integer labels for training images 

# I am calculating the weight for each class
weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights)) # basically converting the weights array to a dictionary


#Build the Convolutional Neuron Network as a stack of layers

def build_cnn(input_shape=(128, 128, 3), num_classes = 4):
    model = Sequential([
        
        #Block 1
        Conv2D(32, (3, 3), activation = 'relu', padding = 'same',
            input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size = (2,2)),


        #Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Classifier head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 4-class output
    ])
    return model


model = build_cnn()
model.summary()


#Compile the model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Callbacks

#This block of code watches the validation loss after every epoch, waits for 10 consecutive epoch
#with no improvement before stopping and prints a message when ealry stopping triggers.
callbacks=[
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        'best_cnn_model.h5', #filename to save the model to
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1

    )
]



history = model.fit(
    train_gen,
    epochs=50, #Early stoppinf will cut this short if model stops improving
    validation_data=val_gen,
    class_weight=class_weights, #This handles imbalance
    callbacks=callbacks,
    verbose=1
)

#Plot Training Curves to make sure the model generalizes well and its not overfitting/underfitting

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history['loss'],     label='Training loss')
axes[0].plot(history.history['val_loss'], label='Validation loss')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy curve
axes[1].plot(history.history['accuracy'],     label='Training accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation accuracy')
axes[1].set_title('Accuracy over epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('cnn_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


#Evaluate on Test Set

# Load the best saved model
from tensorflow.keras.models import load_model
best_model = load_model('best_cnn_model.h5')

# Get predictions
y_pred_proba = best_model.predict(test_gen)
y_pred  = np.argmax(y_pred_proba, axis=1)
y_true  = test_gen.classes

class_names = list(test_gen.class_indices.keys())

# All metrics 
print("=" * 50)
print("CNN TEST SET RESULTS")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=class_names))

# Individual metric extraction for comparison table
acc       = accuracy_score(y_true, y_pred)
macro_f1  = f1_score(y_true, y_pred, average='macro')
macro_pre = precision_score(y_true, y_pred, average='macro')
macro_rec = recall_score(y_true, y_pred, average='macro')

print(f"\nAccuracy:          {acc:.4f}")
print(f"Macro F1-score:    {macro_f1:.4f}")
print(f"Macro Precision:   {macro_pre:.4f}")
print(f"Macro Recall:      {macro_rec:.4f}")


#Confusion Matrix

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Simple CNN — Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


results = {
    'Model':     'Simple CNN',
    'Accuracy':  round(acc, 4),
    'Macro F1':  round(macro_f1, 4),
    'Macro Precision': round(macro_pre, 4),
    'Macro Recall':    round(macro_rec, 4)
}

pd.DataFrame([results]).to_csv('cnn_results.csv', index=False)

print(results)
