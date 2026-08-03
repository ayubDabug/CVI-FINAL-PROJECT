"""
train.py
--------
Train the Nvidia CNN to predict steering angle from a front-camera image.

Steps:
  1. load the combined driving log
  2. plot the raw steering histogram (before balancing)
  3. balance the steering distribution and plot it again (after)
  4. split into train / validation
  5. train with the augmenting batch generator
  6. plot the loss curves
  7. save model.h5

Run (after combine_data.py has produced the combined CSV):
    python train.py --csv data/driving_log_combined.csv --data-dir data \
                    --epochs 15 --batch-size 100
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import (load_dataset, balance_data, plot_histogram,
                   batch_generator, build_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/driving_log_combined.csv')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--steps-per-epoch', type=int, default=300)
    ap.add_argument('--val-steps', type=int, default=200)
    ap.add_argument('--num-bins', type=int, default=25)
    ap.add_argument('--samples-per-bin', type=int, default=400)
    ap.add_argument('--out', default='model.h5')
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. load ----------------------------------------------------------------
    image_paths, steerings = load_dataset(args.csv, args.data_dir)
    if len(image_paths) == 0:
        raise SystemExit("No usable samples found. Check --csv and --data-dir.")

    # 2. histogram before balancing -----------------------------------------
    plot_histogram(steerings, args.num_bins, 'Steering distribution (raw)',
                   args.samples_per_bin,
                   os.path.join(args.outdir, 'hist_before.png'))

    # 3. balance + histogram after ------------------------------------------
    keep = balance_data(steerings, args.num_bins, args.samples_per_bin)
    image_paths, steerings = image_paths[keep], steerings[keep]
    plot_histogram(steerings, args.num_bins, 'Steering distribution (balanced)',
                   args.samples_per_bin,
                   os.path.join(args.outdir, 'hist_after.png'))

    # 4. split ---------------------------------------------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=42)
    print(f"train samples: {len(x_train)}   val samples: {len(x_val)}")

    # 5. train ---------------------------------------------------------------
    model = build_model()
    model.summary()

    history = model.fit(
        batch_generator(x_train, y_train, args.batch_size, is_training=True),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=batch_generator(x_val, y_val, args.batch_size, is_training=False),
        validation_steps=args.val_steps,
        verbose=1,
    )

    # 6. loss curves ---------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Training / validation loss (MSE)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(args.outdir, 'loss_curves.png')
    plt.savefig(loss_path, dpi=120)
    plt.close()
    print(f"saved {loss_path}")

    # 7. save ----------------------------------------------------------------
    model.save(args.out)
    print(f"saved trained model -> {args.out}")


if __name__ == '__main__':
    main()
