"""
Train a CNN (Nvidia end-to-end architecture) to predict steering angle
from the center-camera images collected in ./data recorded.

Usage:
    python train.py --datadir "data recorded" --epochs 10
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from data_loader import load_data, balance_data
from batching import batch_generator
from preprocessing import IMG_WIDTH, IMG_HEIGHT


def nvidia_model():
    """CNN architecture from Figure 7 of the assignment (Nvidia end-to-end model)."""
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation="elu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation="elu"),
        Conv2D(48, (5, 5), strides=(2, 2), activation="elu"),
        Conv2D(64, (3, 3), activation="elu"),
        Conv2D(64, (3, 3), activation="elu"),
        Flatten(),
        Dense(1164, activation="elu"),
        Dense(100, activation="elu"),
        Dense(50, activation="elu"),
        Dense(10, activation="elu"),
        Dense(1),
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data recorded",
                         help="Folder containing driving_log.csv and IMG/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=300)
    parser.add_argument("--val_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--samples_per_bin", type=int, default=400)
    parser.add_argument("--out", default="model")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 3. Load data
    image_paths, steerings = load_data(args.datadir)

    # 4. Review / balance the dataset (also saves before/after histograms)
    image_paths, steerings = balance_data(
        image_paths, steerings, samples_per_bin=args.samples_per_bin, save_dir=args.out
    )

    # Train / validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=6
    )
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_valid)}")

    # 6/7. Build + train model using batch generators
    # (augmentation is applied only to the training generator, not validation)
    model = nvidia_model()
    model.summary()

    history = model.fit(
        batch_generator(X_train, y_train, args.batch_size, is_training=True),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=batch_generator(X_valid, y_valid, args.batch_size, is_training=False),
        validation_steps=args.val_steps,
        verbose=1,
    )

    # Save model
    model_path = os.path.join(args.out, "model.h5")
    model.save(model_path)
    print(f"Saved trained model to {model_path}")

    # Plot + save training/validation loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.title("Training progress")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    loss_plot_path = os.path.join(args.out, "loss.png")
    plt.savefig(loss_plot_path)
    print(f"Saved loss curve to {loss_plot_path}")


if __name__ == "__main__":
    main()
