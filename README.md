# CSDI-Blue Noise

This is a modified version of CSDI to add a blue noise schdule to the standard noise process. I added a lot of comments to help identify the new changes

The only file that was modified was main_model.py in the CSDI folder. 

Modifications:
1. gen_bn.py is used to precompute blue noise for the main process to use. This is not fully refined as the paper does not explain exactly how they did this part.
   
2. The next modification is to use a rectified mapping for the noise. The original paper mentioned using a flow based matching however this lead to major overfitting so I went with a mapping based off hungarian assignment which performed much better

3. The blending of Gaussian noise and blue noise is time dependent based on a gamma function which is added to calculate a blending factor based on the time step sampled.

4. The foward and reverse process where modified to combine the Gaussian noise to the 2D blue noise based off of the gamma blending function. they were also modified to apply the rectified mapping to the blended noise. The rest of the foward and reverse process where left the same.

5. there was no modification to any other parts of the model so the effects on the performance could be directly associated to the addition of the blue noise scheduler. 

6. I also was playing around with using a precomputed Feature X Time covariance decay matrix to sample from instead of standard blue noise, This worked quite well.
