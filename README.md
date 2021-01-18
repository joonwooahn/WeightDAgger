# Weight-DAgger: Reduce DAgger Iteration By Using Action Discrepancy As Weight in Training
Repository to store imitation learning that runs on CARLA simulator.

Requirements
-------
Keras (2.2.4)

tensorflow-gpu (1.13.1)
  
numpy (1.18.2)
  
scipy (1.2.0)

python 2, 3 (3.5 < )
  
CARLA 0.9.7

ros (kinetic)  

opencv-python (4.2.0.32)

pyparsing (2.0.3)

scikit-image (0.10.1)


Running
-------

![system](https://user-images.githubusercontent.com/35481431/104888412-0fda1a00-59b0-11eb-90ae-de4227489d4b.png)

To collect the data and play the trained policy, CARAL and ROS have to be runned.

$ roscore

$ play_carla (cd ./carla/LinuxNoEditor_weight_DAgger_MAP && ./CarlaUE4.sh) 

(If this command is typed at the 'bashrc', you can command it eaisly trun on the CARLA simulator.)

(Ex. alias play_carla='cd && cd /xxx/carla/LinuxNoEditor_ver3 && ./CarlaUE4.sh)

First, you have to run the 1. Behavior Cloning section.

The data collection is process with './weight_DAgger $ python3 dataCollect.py --iter BC'.

Then, the dataset, seg.npy and label.npy are created at './DATA/BC/'

Using these dataset are used to train the policy. ./weight_DAgger $ python3 trainPolicy.py --policy BC

The trained policy 'trained_policy.hdf5' is saved at './RUNS/BC/'.

After train the policy with Behavior Cloning, DAgger (Dataset Aggregation) is processed.

DAgger has two types: 2-1. EnsembleDAgger and 2-2. WeightDAgger (proposed).

The running method of these are the same as the Behavior Cloning section.

* But, 2-1. EnsembleDAgger and 2-2. WeightDAgger are runned repeatedly, untill the desired policy is obtained.

** In the proposed method 2-2. WeightDAgger, the weight update process have to runned between the process of the data collection and the trainning.


Dataset
-------

Trained Policy
-------
