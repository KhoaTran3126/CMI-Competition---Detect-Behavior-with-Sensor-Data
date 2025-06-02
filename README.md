# CMI Competition: Detect Behavior with Sensor Data
Predicting Body Focused Repetitive Behaviors (BFRB) from a Wrist-Worn Device.

## Problem Statement: 
To distinguish BFRB-like activities from non-BFRB-like activities using wrist-worn sensor data on participants, and determine the specific type of BFRB-like gesture. The goal is to determine whether the addition of extra temperature sensors and proximity sensors was worth it? Can we improve detection of BFRB actions more accurately compared to a standard IMU device, opening more development toward how to better understand and treat BFRB?

## Background: 
  Body Focused Repetitive Behaviors (BFRBs): Repetitive actions on body that can cause physical pain/psychosocial challenges. Observable in those with OCD/anxiety disorder and are indicative of mental health challenges. 

## Device used: 
Custom Helios wrist-worn device with 1 IMU, 5 additional temperature sensors (thermopiles), and 5 time-of-flight/distance sensors.

## Experimental Details: 
Participants the following series of repeated gestures whilst wearing the device:
1. Transitioning from ‘rest’ location to moving their hand toward the appropriate location
   
   Rest → **Transition**
   
2. Transition is followed by a short pause of no action
   
   Transition → **Pause**
   
3. Performs a gesture EITHER a BFRP-like action or non-BFRP-like action
   
   Pause → **Gesture** (BFRP or not)

**Each participant performed 18 unique gestures** (**8 BFRB-like gestures + 10 non-BFRB-like gestures**) in at least 1 of 4 different body-positions (sitting, sitting leaning forward with non-dominant arm resting on leg, lying on back, and lying on side)


## Evaluation Metric: 
  **Binary F1** on whether the gesture target or non-target type.
  
  **Macro F1** on gesture, with all non-target sequences collapsed to a single non_target class
  
  **Final score =  (Binary F1 + Macro F1)/2**

## Data type:
  All files are csv files. 
  
## Other Details: 
  Test size contains half of data from IMU sensors, half from custom design sensor.
