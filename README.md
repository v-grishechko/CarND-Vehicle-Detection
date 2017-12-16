## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicles]: ./assets/nonvehicles.png
[nonvehicles]: ./assets/vehicles.png
[hog_visualization]: ./assets/hog_visualization.png
[sliding_window]: ./assets/sliding_window.png
[multiple_sliding_window]: ./assets/multiple_sliding_window.png
[sliding_window_upgrade]: ./assets/sliding_window_upgrade.png
[pipeline]: ./assets/pipeline.png
[video1]: ./project_processed_video.mp4

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second and thirth code cell of the IPython notebook (```pipeline.ipynb```).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicles]
![alt text][nonvehicles]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I tested hog on the vehicle image.
Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_visualization]

There method for extracting hog features:

```python
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and choosed the most efficient combination in term of accuracy of classifying:

| Configuration number	| Colorspace | Orientations | Pixels Per Cell | Cells Per Block | Train Time   | Accuracy 	  |
| :--------------------:| :--------: | :-----------:| :--------------:| :--------------:|:------------:| ------------:|
| 1                  	| RGB        | 9            | 8               | 2               | 15s          | 0.9724       |
| 2                   	| HSV        | 9            | 8               | 2               | 20.04s       | 0.9762       |
| 3                   	| LUV        | 9            | 8               | 2               | 7.85s        | 0.9783       |
| 4                   	| HLS        | 9            | 8               | 2               | 7.74s        | 0.9703       |
| 5                   	| YCrCb      | 9            | 8               | 2               | 6.06s        | 0.9728       |
| 6                  	| YCrCb      | 11           | 8               | 2               | 6.31s        | 0.9762       |
| 7                   	| YUV        | 12           | 8               | 2               | 8.43s        | 0.9777       |
| 8                   	| YUV        | 11           | 8               | 2               | 7.43s        | 0.9689       |
| 9                   	| YUV        | 11           | 16              | 2               | 4s           | 0.9789       |


Code which I used to test different combination of params:
```python
configs = [("RGB", 9, 8, 2),
           ("HSV", 9, 8, 2),
           ("LUV", 9, 8, 2),
           ("HLS", 9, 8, 2),
           ("YCrCb", 9, 8, 2),
           ("YCrCb", 11, 8, 2),
           ("YUV", 12, 8, 2),
           ("YUV", 11, 8, 2),
           ("YUV", 11, 16, 2)]

for config in configs:
    vehicle_features = extract_features(vehicles_files, orient=config[1], pix_per_cell=config[2],
                                        cell_per_block=config[3], hog_channel="ALL", cspace=config[0])
    non_vehicles_features = extract_features(non_vehicles_files, orient=config[1], pix_per_cell=config[2],
                                             cell_per_block=config[3], hog_channel="ALL", cspace=config[0])

    X = np.vstack((vehicle_features, non_vehicles_features)).astype(np.float64)
    Y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicles_features))))

    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("{} {} {} {}".format(config[0], config[1], config[2], config[3]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(train_features, train_labels)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(test_features, test_labels), 4))
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features 

I trained a linear SVM with the default classifier parameters and using HOG features alone (I don't use color and spatial binning features) and achieve accuracy: 97.89%.

This code is contained in sixth cell in IPython book.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used function ```find_cars``` from course, but made in this function some modifications. This function return only rectangles and extract only hog features from all channels of image. Also this function take ```ystart``` and ```ystop``` params (region of hog extracting features).

![alt text][sliding_window]

The dimensions and proportions of the car can vary, so both the size of the window and the search area should change too. I add function ```find_cars_with_search_regions```, which take ```search_regions``` param. 

Example of using different params of searching regions:
![alt text][multiple_sliding_window]

After some experiments I choose this params of search region:

| # | Y start | Y stop | Scale of window |
|:-:|:-------:|:------:|:---------------:|
|1  |  400    | 464    | 1.0			 |
|2  |  416    | 480    | 1.0			 |
|3  |  400    | 496    | 1.5			 |
|4  |  432    | 528    | 1.5			 |
|5  |  400    | 528    | 2.0			 |
|6  |  432    | 560    | 2.0			 |
|7  |  400    | 596    | 3.5			 |
|8  |  464    | 660    | 3.5			 |

Here code of function ```find_cars_with_search_regions```:

```python
search_regions = [(400, 464, 1.0),
                  (416, 480, 1.0),
                  (400, 496, 1.5),
                  (432, 528, 1.5),
                  (400, 528, 2.0),
                  (432, 560, 2.0),
                  (400, 596, 3.5),
                  (464, 660, 3.5)]


def find_cars_with_search_regions(img, svc, orient, pix_per_cell, cell_per_block, cspace, search_regions):
    boxes = []
    for search_region in search_regions:
        boxes.append(find_cars(img, search_region[0], search_region[1], search_region[2], svc, orient, pix_per_cell,
                               cell_per_block, cspace))
    boxes = [item for sublist in boxes for item in sublist]
    return boxes


vehicle_img = mpimg.imread("test_images/test1.jpg")
boxes = find_cars_with_search_regions(vehicle_img, svc, orient, pix_per_cell, cell_per_block, colorspace, search_regions)
plt.figure(figsize=(10, 15))
plt.imshow(draw_boxes(vehicle_img, boxes))
plt.title("Find cars with combinations of search regions")

```

And result of this function:

![alt text][sliding_window_upgrade]
---

### Video Implementation

#### 1. Provide a link to your final video output.
Here's a [link to my video result](./project_processed_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Also I use previous boxes from previous frame to increase accuracy of finding cars in next frames:

```python
boxes = find_cars_with_search_regions(img, svc, orient, pix_per_cell, cell_per_block, colorspace, search_regions)
if len(boxes) > 0:
        prev_boxes_list.append(boxes)
        
    if len(prev_boxes_list) > 13:
        prev_boxes_list = prev_boxes_list[len(prev_boxes_list) - 13:]
```

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][pipeline]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Of course, the algorithm may fail in case of difficult light conditions, weather conditions which could be partly resolved by the classifier improvement. Also, the algorithm merges several cars into one if they close together. It is possible to improve the classifier by additional data augmentation, hard negative mining, classifier parameters tuning etc. The pipeline is not a real-time (about 4 fps with Lane line detection, which independently performs at 9 fps). One can further optimize number of features and feature extraction parameters as well as number of analyzed windows to increase the rate because lane line detection is quite fast.

