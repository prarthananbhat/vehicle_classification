# vehicle_classification
Janata Hack: Emergency vs Non-Emergency Vehicle Classification
1. Total Images : 1646 in Train, 706 in Tes, Emergency Vehicles tagges as 1 and Non emergency Vehicles Tagges as 0
2. Leader board Score : 97.60

Let us look at a few images form the train set.
Trasformation applies are:
1. Horizontal Flip
2. Random Rotation

![raw images](https://github.com/prarthananbhat/vehicle_classification/blob/master/raw%20data.png?raw=true "Raw Data")

Model Traning

1. Traning a restnet 18 model for 2 classes
2. Use Reduce LR on Plateau
3. Loss : Crossentropy
4. Epochs: 30

After training for 30 epochs below is the plots of loss for train and validation set
![loss curves](https://github.com/prarthananbhat/vehicle_classification/blob/master/loss%20curves.png?raw=true "Loss Curves")

View of predictions from validation set
![loss curves](https://github.com/prarthananbhat/vehicle_classification/blob/master/sample%20validations.png?raw=true "Sample Images form Validation Set")

Below are missclassified images from the validation set. "Know what your model has not learnt from this image"
![loss curves](https://github.com/prarthananbhat/vehicle_classification/blob/master/miss%20classified%20images.png?raw=true "Misclassified images")




