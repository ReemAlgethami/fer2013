import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# 1️⃣ Load data
data_path = r"C:\Users\ralfa\OneDrive\fer2013\fer2013_dataset_enhanced.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"⚠ File not found: {data_path}")

df = pd.read_csv(data_path)
print(f"📊 Dataset loaded: {df.shape}")
print(f"Emotion distribution:\n{df['emotion'].value_counts().sort_index()}")
print(f"Usage distribution:\n{df['Usage'].value_counts()}")

# 2️⃣ Verify pixel validity
def verify_pixels(df):
    pixel_lengths = df['pixels'].apply(lambda x: len(x.split()))
    correct_length = 2304
    incorrect_count = (pixel_lengths != correct_length).sum()
    if incorrect_count > 0:
        print(f"❌ {incorrect_count} images removed (invalid pixels)")
        df = df[pixel_lengths == correct_length].copy()
    return df

df = verify_pixels(df)

# 3️⃣ Split data
train_df = df[df['Usage'] == 'Training'].copy()
public_test_df = df[df['Usage'] == 'PublicTest'].copy()
private_test_df = df[df['Usage'] == 'PrivateTest'].copy()

print(f"Training: {len(train_df)}, PublicTest: {len(public_test_df)}, PrivateTest: {len(private_test_df)}")

# 4️⃣ Process pixels safely
def safe_preprocess_images(df):
    images, valid_indices = [], []
    for idx, pixel_sequence in enumerate(df['pixels']):
        try:
            pixels = pixel_sequence.split()
            if len(pixels) != 2304:
                continue
            image = np.array(pixels, dtype=np.uint8).reshape(48, 48, 1)
            if image.min() >= 0 and image.max() <= 255:
                images.append(image)
                valid_indices.append(idx)
        except:
            continue
    return np.array(images) / 255.0, df.iloc[valid_indices]

X_train, train_df = safe_preprocess_images(train_df)
y_train = train_df['emotion'].values
X_test, test_df = safe_preprocess_images(public_test_df)
y_test = test_df['emotion'].values
X_private, private_df = safe_preprocess_images(private_test_df)
y_private = private_df['emotion'].values

print(f"✅ Processed - Train: {X_train.shape}, Public: {X_test.shape}, Private: {X_private.shape}")

# 5️⃣ Class distribution plot (Train / Public / Private)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.countplot(x=y_train, palette="Set2")
plt.title("Training Set")
plt.xticks(ticks=range(len(emotion_labels)), labels=emotion_labels, rotation=30)

plt.subplot(1, 3, 2)
sns.countplot(x=y_test, palette="Set2")
plt.title("Public Test Set")
plt.xticks(ticks=range(len(emotion_labels)), labels=emotion_labels, rotation=30)

plt.subplot(1, 3, 3)
sns.countplot(x=y_private, palette="Set2")
plt.title("Private Test Set")
plt.xticks(ticks=range(len(emotion_labels)), labels=emotion_labels, rotation=30)

plt.tight_layout()
plt.savefig("class_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# 6️⃣ Handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print(f"⚖️ Class weights: {class_weights}")

# 7️⃣ Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 8️⃣ Build Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1), padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 9️⃣ Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
model_checkpoint = ModelCheckpoint('best_emotion_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# 🔟 Training
print("🚀 Training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=len(X_train)//64,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    class_weight=class_weights,
    verbose=1
)

model.load_weights('best_emotion_model.keras')

# 1️⃣1️⃣ Plot training curves
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1,3,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

if 'lr' in history.history:
    plt.subplot(1,3,3)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# 1️⃣2️⃣ Evaluation
print("📊 Evaluation:")
print("Public Test:")
print(model.evaluate(X_test, y_test, verbose=0))
print("Private Test:")
print(model.evaluate(X_private, y_private, verbose=0))

# 1️⃣3️⃣ Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=emotion_labels))

# 1️⃣4️⃣ Sample predictions
def plot_sample_predictions(X, y_true, y_pred_classes, num_samples=10):
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    plt.figure(figsize=(15,8))
    for i, idx in enumerate(indices):
        plt.subplot(2,5,i+1)
        plt.imshow(X[idx].squeeze(), cmap='gray')
        true_label = emotion_labels[y_true[idx]]
        pred_label = emotion_labels[y_pred_classes[idx]]
        color = 'green' if y_true[idx] == y_pred_classes[idx] else 'red'
        plt.title(f"T:{true_label}\nP:{pred_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_sample_predictions(X_test, y_test, y_pred)
