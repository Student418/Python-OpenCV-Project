There are 3 different open cv projects.



The first project is about detection of number of persons crossing a particular area.

In this project, the number of persons that have crossed the deciding line is detected. Cap variable is used to capture images from the camera. 0 Index number shows that the camera is in-built system camera. If there is an extenal camera used for capturing images then index number must be changed accordingly. Haarcascade classifier is used to detect the face of a person. Captured images are converted into gray scale so that its processing can become easier. In gray scale, each pixcel represent certain intensity of light. Vertical_area is the variable that stores the coordinates of deciding line. Counter is the variable that stores how many persons have crossed the deciding line. At starting its value is set to zero. Whenever a person crosses the line there will increment of one in the counter variable. Pyttsx3 library is used to convert text to speech. So when first peron crosses the line, it speaks the word 'First'. When second person crosses the line, it speaks the word 'Second' and so on.........



The second project is about detection of emotion of a person.

In this project, emotion of a person ,i.e, whether the person is happy or sad or neutral is detected. DeepFace library is used to analyse the expression of face in different emotional circumstances. Emotions like disgust,fear,angry are classified in sad emotional state. Whereas emotions like joy,pleasant,peaceful are classified in happy emotional state. Here, BGR (Blue,Green,Red) color space is used. Color = (255, 0, 0) represents blue. Color = (0, 255, 0) represents green. Color = (0, 0, 255) represents red. Color = (255, 255, 0) represents yellow. Happy emotional state is represented in green color whereas sad emotional state is represented in red color. If the face is not detected properly , then it will display the text 'Error analyzing face'.



The third project is about detection of posture of a person.

In this project, posture of a person ,i.e, Sitting, Namastey, Head Down, Neutral, is detected. Mediapipe library is used to detect the posture of a person. Function 'def detect_faces' is used to analyse inclination angle of face from the captured frames. Function 'def detect_hands' is used to analyse the movement of hands. Time module is imported for two purpose. Firstly it is used to create a detection_threshold of 5 seconds so that when head is down for more than 5 seconds then only it it will indicate head down. Secondly it is used to create a sleep delay in the main loop so that the execution rate of the loop can be effectively controlled. In this case, time.sleep(0.1) indicates a 100 millisecond delay. This means that the loop will pause for 100 milliseconds before the next iteration. This helps to manage the frame rate of video processing and display.


