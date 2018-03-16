Follow the steps below to capture the features, and train the SVM.


1. Put the cloned repository (sensor_stick) in the following location
   assuming the workspace is in home directory of the user:

   cp sensor_stick ~/catkin_ws_1/src

2. Change directory to catkin workspace:

   cd ~/catkin_ws_1

3. Install the dependacies of the project for the ros using rosdep:

   rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y

4. Build the catkin workspace:

   catkin_make

5. Ensure that the following (similar) lines are appropriately present in the
   ~/.bashrc file, and if not, add them appropriately and save the file:

   source /opt/ros/kinetic/setup.bash
   source ~/catkin_ws_1/devel/setup.bash
   export GAZEBO_MODEL_PATH=~/catkin_ws_1/src/sensor_stick/models
   # Commented to avoid conflict with ros environment
   #export PATH="/home/sachin/miniconda3/bin:$PATH"

6. Close the terminal and reopen a new terminal, and change to the catkin workspace:

   cd ~/catkin_ws_1

7. Capture the features:
  7.1.  Run the following command to launch the gaebo environment for training:

        roslaunch sensor_stick training.launch

  7.2.  Open a new terminal, and change directory to catkin workspace, and then start
        the feature extraction process:

        cd ~/catkin_ws_1

        rosrun sensor_stick capture_features.py



        This should start the feature extraction process on 100 random samples for each of the 6 objects.
        Wait for the process to complete.
        After the process is complete the capture_features.py script will end automatically.
        A file named training_set.sav should be created in the catkin workspace directory.
        Close the training.launch process also after this.

8. Train the SVM:
   8.1. Ensure to be in catkin workspace:

        cd ~/catkin_ws_1

   8.2. Train the SVM using the features extracted previously;

        rosrun sensor_stick train_svm.py



        This should create a file model.sav in the catkin workspace, which can be shared.

9. (Optional) If more training features are required, do the following:

   9.1. Open up the following file using a editor:

        vi ~/catkin_ws_1/src/sensor_stick/scripts/capture_features.py


        Change the number of random samples for training to 1000 from 100.
        Save the file.

    9.2. Follow step 7 and 8 to generate the model.sav which can be shared.



      
