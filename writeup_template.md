**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[Hog features with YUV for car on ch0]: ./output_images/hog_features_yuv_car_ch0.jpg
[Hog features with YUV for car on ch1]: ./output_images/hog_features_yuv_car_ch1.jpg
[Hog features with YUV for car on ch2]: ./output_images/hog_features_yuv_car_ch2.jpg
[Hog features with YUV for notcar on ch0]: ./output_images/hog_features_yuv_notcar_ch0.jpg
[Hog features with YUV for notcar on ch1]: ./output_images/hog_features_yuv_notcar_ch1.jpg
[Hog features with YUV for notcar on ch2]: ./output_images/hog_features_yuv_notcar_ch2.jpg
[Hog features with YCrCb for car on ch0]: ./output_images/hog_features_ycrcb_car_ch0.jpg
[Hog features with YCrCb for car on ch1]: ./output_images/hog_features_ycrcb_car_ch1.jpg
[Hog features with YCrCb for car on ch2]: ./output_images/hog_features_ycrcb_car_ch2.jpg
[Hog features with YCrCb for notcar on ch0]: ./output_images/hog_features_ycrcb_notcar_ch0.jpg
[Hog features with YCrCb for notcar on ch1]: ./output_images/hog_features_ycrcb_notcar_ch1.jpg
[Hog features with YCrCb for notcar on ch2]: ./output_images/hog_features_ycrcb_notcar_ch2.jpg
[Boundary boxes and heat map for test_images/test1.jpg]: ./output_images/bbox_hmap1.jpg
[Boundary boxes and heat map for test_images/test2.jpg]: ./output_images/bbox_hmap2.jpg
[Boundary boxes and heat map for test_images/test3.jpg]: ./output_images/bbox_hmap3.jpg
[Boundary boxes and heat map for test_images/test4.jpg]: ./output_images/bbox_hmap4.jpg
[Boundary boxes and heat map for test_images/test5.jpg]: ./output_images/bbox_hmap5.jpg
[Boundary boxes and heat map for test_images/test6.jpg]: ./output_images/bbox_hmap6.jpg
[Heatmap of detections]: ./output_images/heatmap.jpg
[Boundary boxes]: ./output_images/boxes.jpg
[video1]: ./project_video_out.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
HOG features are extracted in get_hog_features() with orient = 9  pix_per_cell = 8 cell_per_block = 2

I first used colorspace=YUV

![Hog features with YUV for car on ch0]: Hog features with YUV for car on ch0

![Hog features with YUV for car on ch1]: Hog features with YUV for car on ch1

![Hog features with YUV for car on ch2]: Hog features with YUV for car on ch2

![Hog features with YUV for notcar on ch0]: Hog features with YUV for notcar on ch0

![Hog features with YUV for notcar on ch1]: Hog features with YUV for notcar on ch1

![Hog features with YUV for notcar on ch2]: Hog features with YUV for notcar on ch2


I switched to colorspace=YCrCb as the results for detection were much better. More explanation in #2 below. 
![Hog features with YCrCb for car on ch0]: Hog features with YCrCb for car on ch0

![Hog features with YCrCb for car on ch1]: Hog features with YCrCb for car on ch1

![Hog features with YCrCb for car on ch2]: Hog features with YCrCb for car on ch2

![Hog features with YCrCb for notcar on ch0]: Hog features with YCrCb for notcar on ch0

![Hog features with YCrCb for notcar on ch1]: Hog features with YCrCb for notcar on ch1

![Hog features with YCrCb for notcar on ch2]: Hog features with YCrCb for notcar on ch2



#### 2. Explain how you settled on your final choice of HOG parameters.
I experimented a lot back and forth with feature extraction HOG parameters: 
I Started with color_space = 'YUV', orient = 9  pix_per_cell = 8 cell_per_block = 2 spatial_size (8, 8), hist_bins 8. The detection on test images was not so great so I decided to increase the spatial size to 16,16 and hist_bins to 16. After experimenting more, I increased the pix_per_cell=16 and normalization of cell_per_block to 4. This gave me a reduced feature set that worked well in images but not soo great on the video clip. 

Next I switched to color_space = 'YCrCb' and increased the spatial size to 32, 32 and hist_bins to 32 (like in the lecture). Extracting more color information helped with detection on test images and video clip. With pix_per_cell=16 the detection was still just okay not too great. So I reduced the hog feature extraction cell size to 8x8 with normalization over cell_per_block=2. The feature extrraction vector increased dramtically in size but with these set of features the detection was the best on the test images and the video clip

Final set of feature extraction parameters:
color_space = 'YCrCb'
hog_channel = 'ALL'
orient = 9  
pix_per_cell = 8 
cell_per_block = 2  

spatial_size = (32, 32)
hist_bins = 32


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
I trained a linear SVM classifier using training data for cars and not car images provided as part of the training set. I extracted spatial, color histogram and HOG features from the training data set and ran it on the test data set. 

The images were split in a 80% training data set and 20% test data set using train_test_split(). The data used for training was shuffled to avoid overfitting so that a sequence of images are not identical from each other. 

The classifier is implemented in cell noted as "# Train the classifier"

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I implemented the sliding window search in the find_cars(). I chose to extract features and make predictions on car or not car using hog sub-sampling once rather than extracting hog features from each overlapping sliding window. The hog sub-sampling method is far more efficient than extracting hog features over each overlapping windows. 

The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. A window of 8x8 cells with cells_per_step=2 gives an overlap of 75%. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
I started with scale 1.5 but noticed that the detection worked a lot better with scale=2 using YCrCb colorspace 3-channel HOG features along with spatially binned color and histograms of color features in a normalized feature vector.  Here's the output of the pipeline on the test images:

![Boundary boxes and heat map for test_images/test1.jpg]: Boundary boxes and heat map for test_images/test1.jpg

![Boundary boxes and heat map for test_images/test2.jpg]: Boundary boxes and heat map for test_images/test2.jpg

![Boundary boxes and heat map for test_images/test3.jpg]: Boundary boxes and heat map for test_images/test3.jpg

![Boundary boxes and heat map for test_images/test4.jpg]: Boundary boxes and heat map for test_images/test4.jpg

![Boundary boxes and heat map for test_images/test5.jpg]: Boundary boxes and heat map for test_images/test5.jpg

![Boundary boxes and heat map for test_images/test6.jpg]: Boundary boxes and heat map for test_images/test6.jpg

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a ![link to my video result](./project_video_out.mp4)

Here's my video result ![video1]:


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes. 
In process_image() I accumulate boundary box detections over n frames and set a high threshold such that the correct detections get a good hit on the heat map and the false positives filter out as they are below the threshold. 

I found that for a single image setting the threshold to 2 for plotting the boundary box on the heatmap was good. Then I scaled that to average over 50 frames (approximately 2 seconds of video clip) and used the same detection to heatmap ratio of 1:2 that I discovered through the test images. This allowed me to hold a correct detection longer and filter out false positives.  

Here's an example boundary boxes and heatmaps from the last frame of 10-12 seconds of the video clip:
![Heatmap of detections]: Heatmap of detections

![Boundary boxes]: Boundary boxes


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
1. The pipeline fails when the car is at a distance like at around 25 through 29 seconds of the video
2. The cars moving on the opposite side of the road are detected and cause false positives sometimes. They should be filtered out somehow 

