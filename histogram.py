import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\Kaung\OneDrive\Desktop\CVI"
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")

columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
data = pd.read_csv(CSV_PATH, names=columns)

print(f"Total samples: {len(data)}")

def path_leaf(path):
    return os.path.basename(path.strip())

data['Center'] = data['Center'].apply(path_leaf)
data['Left']   = data['Left'].apply(path_leaf)
data['Right']  = data['Right'].apply(path_leaf)

num_bins = 25
samples_per_bin = 1000

hist, bins = np.histogram(data['Steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5

plt.bar(center, hist, width=0.05, color='#6699cc')
plt.plot((np.min(data['Steering']), np.max(data['Steering'])),
         (samples_per_bin, samples_per_bin), color='#6699cc')
plt.ylim(0, 1000)
plt.xlabel('Steering Angle')
plt.ylabel('Count')
plt.title('My Steering Angle Distribution')
plt.show()