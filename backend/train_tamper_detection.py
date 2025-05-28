import os, io, argparse
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_SIZE    = (128, 128)   
TRAIN_QUALITY = 90          
BATCH_SIZE    = 32
EPOCHS        = 30
PATIENCE      = 5
VALID_INPUT_EXTS = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')

def make_ela(path, quality=TRAIN_QUALITY, size=TRAIN_SIZE):
    img = Image.open(path).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    rec = Image.open(buf)
    diff = ImageChops.difference(img, rec)
    maxd = max(e[1] for e in diff.getextrema()) or 1
    ela = ImageEnhance.Brightness(diff).enhance(255.0/maxd)
    return ela.resize(size)


def generate_ela_dataset(src, dst):
    """Convert raw images to ELA and save under dst/{authentic,tampered}"""
    for label in ('authentic','tampered'):
        in_dir  = os.path.join(src, label)
        out_dir = os.path.join(dst, label)
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(in_dir):
            if not fname.lower().endswith(VALID_INPUT_EXTS): continue
            src_path = os.path.join(in_dir, fname)
            base,_   = os.path.splitext(fname)
            out_path = os.path.join(out_dir, base + '.jpg')
            try:
                ela_img = make_ela(src_path)
                ela_img.save(out_path, format='JPEG')
            except Exception as e:
                print(f"Skipping {src_path}: {e}")


def build_model(input_shape=(*TRAIN_SIZE,3)):
    """
    Simple CNN as per paper:
    - Conv2D(32,5x5) + Conv2D(64,5x5) + MaxPool + Dropout
    - Flatten + Dense(256) + Dropout + Dense(2)
    """
    m = Sequential([
        Conv2D(32, (5,5), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (5,5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    m.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m


def plot_history(history, prefix):
    """Plot and save loss/accuracy curves."""
    epochs = range(1, len(history.history['loss'])+1)
    plt.figure(figsize=(12,5))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, history.history['loss'], label='train')
    plt.plot(epochs, history.history['val_loss'], label='val')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, history.history['accuracy'], label='train')
    plt.plot(epochs, history.history['val_accuracy'], label='val')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{prefix}_history.png')
    plt.show()


def evaluate(model, gen, prefix):
    """Evaluate model: print report and plot confusion matrix."""
    y_true = gen.classes
    y_pred = np.argmax(model.predict(gen), axis=1)
    labels = list(gen.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white')
    plt.tight_layout()
    plt.savefig(f'{prefix}_cm.png')
    plt.show()


def main(args):
    print("1) Generating ELA dataset…")
    generate_ela_dataset(args.raw_data, args.ela_data)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        args.ela_data, target_size=TRAIN_SIZE, batch_size=BATCH_SIZE,
        subset='training', class_mode='sparse', shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        args.ela_data, target_size=TRAIN_SIZE, batch_size=BATCH_SIZE,
        subset='validation', class_mode='sparse', shuffle=False
    )

    print("2) Building model…")
    model = build_model()
    callbacks = [
        EarlyStopping('val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(args.output_model, save_best_only=True, monitor='val_loss')
    ]

    print("3) Training…")
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS, callbacks=callbacks
    )

    print("4) Plotting & evaluating…")
    plot_history(history, 'results')
    evaluate(model, val_gen, 'results')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data', required=True, help='raw/{authentic,tampered}')
    parser.add_argument('--ela-data', required=True, help='where to save ELA images')
    parser.add_argument('--output-model', default='models/cnn_tamper.h5', help='save best model')
    main(parser.parse_args())