# CSDI-Blue Noise

This is a modified version of CSDI to add a blue noise schdule to the standard noise process. 

The only file that was modified was main_model.py in the CSDI folder. 

Modifications:
1. The first modification was to initialize a 128x128 Covarience matrix by time and features through Cholesky decomposition. A tiling approach is used to deal with the large memory constraints of the full data set size.

2. I also added code to create noise from only the time dimension rather than time and features however I did not use it in the final model. Its possible a blend of this and the 2D matrix would make sense.

3. The next modification is to use a rectified mapping for the noise. The original paper mentioned using a flow based matching however this lead to major overfitting so I went with a mapping based off hungarian assignment which performed much better

4. The blending of Gaussian noise and blue noise is time dependent based on a gamma function which is added to calculate a blending factor based on the time step sampled.

5. The foward and reverse process where modified to combine the Gaussian noise to the 2D blue noise based off of the gamma blending function. they were also modified to apply the rectified mapping to the blended noise. The rest of the foward and reverse process where left the same.

6. there was no modification to any other parts of the model so the effects on the performance could be directly associated to the addition of the blue noise scheduler. 
