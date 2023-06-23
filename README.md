# Vision System

Using libraries such as OpenCV, Keras and Tensorflow its possible to detect and recognize various objects.

Model used in application was trained on objcets such as dark glass, white frame, strip, and clip.

Using the OpenCV library, the edge, element position and element oreientation are extracted.<br>Based on this, a portion of frame captured by cam is cropped and presented to neutral network for object recognition.

![Object detection](https://github.com/SzyJar/VisionSystem/assets/107247457/72449434-a512-426e-bf70-fb7dba7b976c)

It is also possible to manually set the desired position and orientation of an item by clicking on it on the screen.<br>

![Object calibration](https://github.com/SzyJar/VisionSystem/assets/107247457/0d1444e0-1cdc-4b32-bf22-aaa30befd0ba)

With the right configuration, edge detection can work in a variety of scenarios. As long as the object is detected correctly, the neutral network can predict a category to the detected object.

![Edge detection](https://github.com/SzyJar/VisionSystem/assets/107247457/2feceb10-8ec5-419e-8676-39aa28e720e4)
