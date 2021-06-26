# GANTheftAutoWithAOgmaNeo

These are the training scripts for training an AOgmaNeo model on Sentdex's "GAN Theft Auto" dataset.
Sentdex's original repository, which includes the sample dataseet, is [here](https://github.com/Sentdex/GANTheftAuto).

These scripts assume [PyAOgmaNeo](https://github.com/ogmacorp/PyAOgmaNeo) is installed, as well as OpenCV.

To use, make sure the dataset is in a folder called "gtagan_2_sample" within this folder, and then run neoTrainer.py.
It will periodically save out the training progress when it says "Saved", after which you can run neoPlayer.py to run the result.
