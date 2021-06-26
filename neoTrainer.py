import pyaogmaneo as neo
import numpy as np
import pickle
import gzip
import os
import random
import cv2

class GTADataset:
    def __init__(self, datadir=''):
        self.samples = []
        files = os.listdir(datadir)
        num_data = len(files)

        sample_list = files[:int(num_data * 0.9)]

        for f in sample_list:
            path = os.path.join(datadir, f)
            self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the sequence
        with gzip.open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        # Initialize data lists and calculate the episode length
        states, actions = [], []

        ep_len = len(data['observations'])

        # Iterate over num_steps steps
        for i in range(ep_len):
            cur_s = data['observations'][i]
            cur_a = data['actions'][i]

            # Add to the lists
            states.append(cur_s)
            actions.append(cur_a)

        # Return data
        return states, actions

def main():
    load = False

    ds = GTADataset(datadir='gtagan_2_sample')

    hiddenSize = (80, 48, 16)
    imgSize = (80, 48, 3)
    numActions = 3

    neo.setNumThreads(8)

    lds = []

    for _ in range(5):
        ld = neo.LayerDesc()
        ld.hiddenSize = (8, 8, 16)
        ld.eRadius = 2
        ld.dRadius = 2
        ld.ticksPerUpdate = 2
        ld.temporalHorizon = 2

        lds.append(ld)

    enc = neo.ImageEncoder()
    h = neo.Hierarchy()

    if load:
        enc.initFromFile("enc.oenc")

        h.initFromFile("h.ohr")

        print("Loaded hierarchy.")
    else:
        enc.initRandom(hiddenSize, [ neo.ImageEncoderVisibleLayerDesc(imgSize, 1) ])

        h.initRandom([ neo.IODesc(hiddenSize, eRadius=6, dRadius=2), neo.IODesc((1, 1, numActions), eRadius=0, dRadius=2) ], lds)

        print("Created random hierarchy.")

    print("Pre-training...")

    # Pre-train image encoder, probably uneccessary
    preTrainIters = 1000

    for it in range(preTrainIters):
        idx = random.randint(0, len(ds) - 1)

        states, actions = ds[idx]

        for _ in range(100):
            t = np.random.randint(0, len(states))

            img = cv2.resize(states[t], (imgSize[0], imgSize[1]))
            img = np.swapaxes(img, 0, 1)

            enc.step([ img.ravel().tolist() ], True)

        print(f"It{it}/{preTrainIters}")

    print("Pre-training complete.")

    # Train the hierarchy
    for it in range(100000):
        idx = random.randint(0, len(ds) - 1)

        states, actions = ds[idx]

        for t in range(len(states)):
            img = cv2.resize(states[t], (imgSize[0], imgSize[1]))
            img = np.swapaxes(img, 0, 1)

            enc.step([ img.ravel().tolist() ], True)
            h.step([ enc.getHiddenCIs(), [ actions[t] ] ], True)

            if t % 100 == 0:
                print(f"T{t}")

        print(f"It{it}")

        if it % 10 == 0:
            enc.saveToFile("enc.oenc")
            h.saveToFile("h.ohr")
            print("Saved.")
    
if __name__ == '__main__':
    main()
