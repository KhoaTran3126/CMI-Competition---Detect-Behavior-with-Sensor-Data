# Week 1 Progress

## General Questions
1. How can demographics data be incorporated into train? Can I utilize distance measures in demographics data to enhance my predictions? Does handedness matter?
2. What are the relationships of each feature to the prediction of gesture?
3. Can I condense information of acceleration, temperature, time-of-flight, and rotation sensors' data? 
4. Amongst other predictive features, **orientation**, **phase**, and **behavior** features are **not present in test set**. Are they predictive of gesture at all? Is it possible to infer them? How?
5. What can I use **sequence_counter** for, aside that it indicates a new subject's action?

## Hypotheses
1. **acc_**, **rot_**, and **tof_v_** features can encode **orientation**, **phase**, **behavior** features by clustering.
2. Only summary statistics of the **last 20% of sensors data per sequence** are useful for detecting gestures. An indicator for the tail of the sequence would be useful, and its interactions with other numeric features as well.   

## Discoveries
The hierachy of the data follows the outline of subject-sequence, where each subject generates 102 sequences typically, composing of 64 target and 38 non-target sequences. Certain summaries of sensor distributions can provide slight separations between classes. 

# Next Steps
1. Test these tuned baseline classifiers: LGBM, CatBoost, XGB
2. Test ensemble method
3. Try projection to understand patterns in data

