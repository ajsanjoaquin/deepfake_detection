# Training and Visualization on FaceForensics++

## Experimental Scripts
**main.py** - Standard training and testing
**generate_gradients.py** - Get numpy array of gradients
**visualize.py** - Visualize normalized gradient image of a given image
**visualize2_attack.py** - Visualize FGSM attack on given images and the corresponding accuracies given a specific epsilon

## Helper Scripts
**move_json.py** - Move videos accoridng to official FF++ Train/Valid/Test splits
**crop.py** - Crop the face from a given frame (100 frames are extracted per video)
**argument.py** - Contains all parameters used. Mainly used for main.py

## Model Scripts
**xception.py and models.py** - Scripts to load xception net

## Randomized Smoothing
**code/train.py** - apply randomized smoothing on xception model by training a classifier on gaussian-agumented images <br>
**code/train_pgd.py** - apply train.py + adversarial training based on Projected Gradient Descent (PGD)
For more information on the scripts: I adapted the scripts from [here](https://github.com/Hadisalman/smoothing-adversarial)