# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. Find jupyter notbook implementation [here](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

---

### Data Set Summary & Exploration

##### 1. Loded picke and convert data to numpy array. Used numpy library to get data summary. The code for this step are contained in second and third code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)".Followings are summary:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

##### 2. For exploratory visualization of the dataset, I have used matplotlib to show example of each class and panda dataframe to show number of example from each class in training, validation and test dataset.The code for this step are contained in fourth, fifth and sixth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)"

### Design and Test a Model Architecture

##### 1. Preprocessing code is contained in the eighth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)".

When model was trained without any preprocessing (except randomaly shuffled) validation accuracy was around 0.89 where training accuracy was 0.999. This means model was overfitting (able to memorize training data). One of the cause of overfitting is less number of training set. We don't have more dataset. Training data of a lot of classes was very low. So it requires to increase training data somehow. If we change training images such a way that it belongs to same class as it was eariler (i.e change brightness, change contrast etc.), virtually we can increase number of traing images. This technique is called augmentation.

Following preprocesses are done:
* Randomally changed brightness of training image.
* Randomally changed contrast of training image.
* Standardization of training, validation and test image.

Augmentation technique also include fliping, rotating by some degree, distortation etc. Why we don't have used this technique in our preprocess? Because fliping and rotating by some degree may change class of image. 

##### 2. Model is LeNet (CNN model). Model code is contained in the nineth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)"

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	 |
| RELU					|												|
| Max pooling	 2x2    	| 2x2 stride,  outputs 14x14x6, valid padding 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	 2x2    	| 2x2 stride,  outputs 5x5x16, valid padding 				|
| Flatten Layer       | output 400        |
| Dropout             | 0.5 probability   |
| Fully connected		| input 400, output=120        									|
| Dropout             | 0.5 probability   |
| Fully connected		| input 120, output=84        									|
| Fully connected		| input 84, output=43       									|
| Softmax				|        									|

 
 ##### 3. Training of model code is contained in the tenth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)". Used AdamOptimizer for softmax loss optimization.
 
 Following are hyperparameters:
 * EPOCHS = 30
 * BATCH_SIZE = 128
 * LEARNING_RATE = 1e-5
 
 ##### 4. Model parameters are being saved only if validation loss reduced. Model is LeNet model. When train without augmentation and dropout validation accuracy was arround 0.87. After adding augmentation validation accuracy increased to 0.93 and train accuracy was 0.99. This implies that there was overfitting. So added dropout, which increases validation accuracy to 0.97. Code for calculating accuracy are contained in twelfth and thirteenth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)"
 
 Followings are accuracy summary:
 * Train accuracy:	    0.997 
 * Validition accuracy:	0.972
 * Test accuracy:	      0.958
 
 [//]: # (Image References)

[speed_limit_70]: ./examples/speed_limit_70.jpg
[stop]: ./examples/stop.jpg
[priority_road]: ./examples/priority_road.jpg
[roundabout]: ./examples/roundabout.jpg
[general_caution]: ./examples/general_caution.png
 
 ### Test a Model on New Images
 
 ##### 1. Choosen five German traffic signs found on the web.
 
 ![alt text][speed_limit_70] ![alt text][general_caution] 
 ![alt text][priority_road] 
![alt text][roundabout] ![alt text][stop]

##### 2. Accuracy of model on new images are 0.8. Code to find accuracy and top k probabilities are contained in seventeenth and eighteenth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)".

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Priority road     			| Priority road										|
| General caution					| General caution											|
| Roundabout mandatory	      		| Go straight or right					 				|
| Speed limit (70km/h)		| Speed limit (70km/h)      							|


##### 3. Top K Probabilities code is contained in twentieth code cell of the IPython notebook "[Traffic_Sign_Classifier.ipynb](https://github.com/pchandra90/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)".

###### The top five soft max probabilities (round up to two decimal) for image "Roundabout mandatory" as follows:

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.24     		| Go straight or right   									| 
| 0.23    			| Roundabout mandatory										|
| 0.22					| Keep right										|
| 0.08	      		| Turn right ahead				 				|
| 0.06		| End of all speed and passing limits      							|
 
We can see that top 3 prediction are similar and probabilities are also close. Should be notice that none of the probability is very high.

###### The top five soft max probabilities (round up to two decimal) for image "Speed limit (70km/h)" as follows:

| Probability		        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     		| Speed limit (70km/h)  									| 
| 0.00    			| Speed limit (30km/h)									|
| 0.00					| Speed limit (20km/h)										|
| 0.00	      		| Speed limit (60km/h)			 				|
| 0.00		  | No vehicles      							|

Model is highly confident and rightly predicted.

###### The top five soft max probabilities (round up to two decimal) for image "Stop" as follows:

| Probability		        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     		| Stop  									| 
| 0.00    			| Speed limit (60km/h)									|
| 0.00					| Yield										|
| 0.00	      		| Speed limit (80km/h)			 				|
| 0.00		  | Bumpy road      							|

Model is highly confident and rightly predicted.

###### The top five soft max probabilities (round up to two decimal) for image "Priority road" as follows:

| Probability		        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     		| Priority road									| 
| 0.00    			| No entry									|
| 0.00					| No passing for vehicles over 3.5 metric tons										|
| 0.00	      		| Keep right			 				|
| 0.00		  | No passing     							|

Model is highly confident and rightly predicted.

###### The top five soft max probabilities (round up to two decimal) for image "General caution" as follows:

| Probability		        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     		| General caution								| 
| 0.00    			| Traffic signals									|
| 0.00					| Pedestrians										|
| 0.00	      		| Right-of-way at the next intersection			 				|
| 0.00		  | Go straight or left     							|

Model is highly confident and rightly predicted.


