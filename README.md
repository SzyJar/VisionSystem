# Vision System

Using libraries such as OpenCV, Keras and Tensorflow its possible to detect and recognize various objects.

Model used in application was trained on objcets such as dark glass, white frame, strip, and clip.

Using the OpenCV library, the edge, element position and element oreientation are extracted.<br>Based on this, a portion of frame captured by cam is cropped and presented to neutral network for object recognition.

![Zrzut ekranu 2023-05-29 171338](https://github.com/SzyJar/VisionSystem/assets/107247457/0e75a267-60b5-423d-b95b-39ddf6d1a282)

It is also possible to manually set the desired position and orientation of an item by clicking on it on the screen.<br>

![Zrzut ekranu 2023-05-29 171404](https://github.com/SzyJar/VisionSystem/assets/107247457/9bff89e0-781c-458c-9da6-2c2bf2f70a48)

With the right configuration, edge detection can work in a variety of scenarios. As long as the object is detected correctly, the neutral network can predict a category to the detected object.

![Zrzut ekranu 2023-05-29 171609](https://github.com/SzyJar/VisionSystem/assets/107247457/55c1fce3-516a-4dd7-b104-c698bffd3da6)
