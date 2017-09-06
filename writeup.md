**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the image and video pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images and HOG Parameters.

The code for this step is contained in the first code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
I tried various combinations of parameters and finally used the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`
Here is an example of one of each of the `vehicle` and `non-vehicle` classes. 

![alt text](output_images/image1.png)

####2. Training a classifier using my selected HOG features (and color features).

I trained a linear SVM using parameters mentioned above.

###Sliding Window Search

####1. Implementing a sliding window search (Decide what scales to search and how much to overlap windows)

I used the method called find_cars. This method combines HOG feature extraction with a sliding window search. The strategy is to extract HOG features for the entire image or its selected portion, followed by subsampling these full-image features according to the size of the window and classification. The prupose is to save time from feature extraction on each window individually. 

The method returns a list of rectangle objects corresponding to the windows that predicts a car. Here is the exmple of the image:  

![alt text](output_images/image2.png)

####2. Examples of test images (optimizing the performance of my classifier)

Finally I searched on various scales using YUV 3-channel (RGB, HLS, and HSV are also explored) HOG features by combining Various Sliding Window Searches, which provided a nice result.  I used explored several configurations of window sizes and positions, with various overlaps in the X and Y directions, along with try and error method by tuning those parameters (window sizes/scales, overlap ratio) and chose best resuls what I have on a given time.  The combination of orient (11), pix_per_cell (16), cell_per_block (2), hog_channel ('ALL') produced the best results among several tests.  
Lastly, the rectangles returned from each method call are aggregated with a scale of window ranging from 1.0 to 3.5, resulting in least false positives. ystart and ystop are examined and manipulated because only certain range in vertical direction is considered for window size to avoid false positives, meanwhile finishing detection in a reasonable speed. 

The image below shows the rectangles returned by find_cars drawn onto one of the test images, indicating that there are several positive predictions on these two cars on the same direction of neighboring lanes:

![alt text](output_images/image3.png)
---

####3. Implementing filter for false positives and method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is the corresponding heatmaps:

![alt text](output_images/image4.png)

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the test image:
![alt text](output_images/image5.png)

Here the resulting bounding boxes are drawn from all test images:

![alt text](output_images/image7.png)


### Video Implementation

####1. Store information

Because the sliding windows are changing sizes on the test images, there is one class added to save information from previous frame in the pipeline so that the detection on the video is more consistent in each frame of the video.
My pipeline performs reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes but identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


###Discussion
####1. Problems / issues  
When the car emerges from the field of view, there are short periods that cars are not detected at the first time. When two cars appeared side by side, there are single detection rather than two detections. There is still one false positve.

I think the problems come from less datasets. If time allowed, augmenting data can be implmented to improve prediction. Other steps can be considered too, such as choice of feature vector, thresholding the decision function, hard negative mining etc. 

Here is an example of augmentation to try:

def create_variant(image):
    if (random.choice([1, 0])):
        image = scipy.ndimage.interpolation.shift(image, [random.randrange(-3, 3), random.randrange(-3, 3), 0])
    else:

        image = scipy.ndimage.interpolation.rotate(image, random.randrange(-8, 8), reshape=False)
    return image

 

