Based on that [project](https://github.com/abdullahtarek/football_analysis)


# Introduction
The project receives a football video, and tracks the players, the referees and the ball. The detections are made using a customly trained YOLO model on that [dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc). 
From the trackings we extract information for which player has the possesion of the ball on each frame. Further, we assign each player on a team based on the t-shirt color allowing us to calculate the total team possesion. Further information can be extracted such as the speed of each player. 
The output of the project is a video that has all extra infromation depicted on each frame. Improvement were made on the original project to making it faster and using less memory 

# Modules 
* YOLO: A YOLO model trained on football annotated pictures is initialy used to make bounding box predictions for players, referees and the ball on each frame.
* OpenCV: It is used for the processiog of the input frames and the creation of the output frames. Also, we use Optical Flow to calculate the camera movement and Perspective Transformation to get the true size of the pitch.
* Scikit-Learn: A K-Means model was used to find the team's colors and assign each player to a team

# Project Parts
* [Tracker](tracker.py): Gets detections from a YOLO model, turns them into trackings and stores them in a numpy format
* [Assigning Teams](team_assigning.py): Uses a K-Means model on player images to find their t-shirt color and assign them to a team
* [Camera Movement](camera_movement.py): Calculates the camemra movement using Optical Flow from CV2 and adjusts the players' positions based on that
* [Perspective Transformer](view_transformer.py): Uses a Perspective Transformer from CV2 to take into consideration the camera perspective for the player positions
* [Speed](speed_estimator.py): Calculates the speed of each player based on distance covered on recent frames
* [Ball Posession](ball_assigner.py): For each frame it finds the closest player to the ball and assigns the possesion. Based on the player's team we can calculate totat team possesion.
* [Drawing](drawing.py): Creating the output frames by adding all the informations gathered on the previous steps

# Improvements
My implementation of the original project was crashing on Google Colad as it was exceeding the RAM limitation for the free-plan (12 GB). This version can run from start to end on the free plan limitations. 

The main source of improvement over the original project is the storing and handling of the tracking infromation on each step of the processing. Instead of using a dict to store the data and store it to stub files, this version uses numpy format which creates more lightweight files that are faster to save and load. 
Furthermore the numpy format allow us to remove a lot of loops to get through the data by using masks instead making the code faster. Additionaly, this version replaces the class-based implementations with function-based implementations. 

