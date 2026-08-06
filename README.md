# TolTEC_pointing
A set of notebooks meant to aid TolTEC researchers in aligning individual observations and correcting coadded image positioning.

## align_observations.ipynb
This notebook finds the local Alt/Az offsets necessary to align individual observations into a coherent final coadded image. These offsets are found via Enhanced Cross Correlation Maximization, which figures out the best x-y offsets with which to align each observation to a reference observation. These offsets should be inputted into the associated .yaml file in Tolteca under pointing_offsets.

## correct_position.ipynb
This notebook makes a necessary positioning correction to the final coadded image. This is done by choosing a point source (VLA 1623-243 in Rho Oph A is given as an example), and comparing its actual position to what the image thinks its position is. The source's position in the image is found via iterative centroiding.  The corrected image will have a '_centered' added to it.

## offset_functions.py
This python file contains the two most vital functions in the above notebooks: relative_pointing_offsets and absolute_pointing_offset. They should be looked at and scrutinized if any errors occur.
