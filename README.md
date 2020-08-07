# Training and Visualization on FaceForensics++ (FF++)

## Experimental Scripts
**main.py** - Standard training and testing <br>
**generate_gradients.py** - Get numpy array of gradients <br>
**visualize.py** - Visualize normalized gradient image of a given image <br>
**visualize2_attack.py** - Visualize FGSM attack on given images and the corresponding accuracies given a specific epsilon <br>

## Helper Scripts
**move_json.py** - Move videos accoridng to official FF++ Train/Valid/Test splits <br> 
**crop.py** - Crop the face from a given frame (100 frames are extracted per video) <br>
**argument.py** - Contains all parameters used. Mainly used for main.py <br>

## Model Scripts
**xception.py and models.py** - Scripts to load xception net <br>

## Randomized Smoothing
**code/train.py** - apply randomized smoothing on xception model by training a classifier on gaussian-agumented images <br>
**code/train_pgd.py** - apply train.py + adversarial training based on Projected Gradient Descent (PGD) <br>
For more information on the scripts: I adapted the scripts from [here](https://github.com/Hadisalman/smoothing-adversarial)

## Sample Job Scripts
I provide 4 sample scripts that I used for training and testing. <br>
1. **M3_train.pbs** - Train a classifier on cropped faces <br>
2. **grad.pbs** - generate the gradient arrays (in .npy format) using best checkpoint from #1 <br>
3. **M4_train.pbs** - Train a classifier on gradient arrays <br>
4. **m4_gradtest.pbs** - Test M4 classifier (works as a generic template for testing too) <br>
5. **smoothed_train.pbs** - Train a robust classifier on Images applied with the Randomized Smoothing method <br>
