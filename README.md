# TolTEC_pointing
A set of notebooks meant to aid TolTEC researchers in aligning individual observations and correcting coadded image positioning.

## How to Use
To properly use this code, first download the entire TolTEC_pointing folder and move it into a directory in Jupyter. You can download a desired reduction folder from Unity into the Jupyter directory by navigating to the directory on the terminal and running "scp -ri ~/.ssh/unity_toltec_guest_umass_edu_only toltec-guest_umass_edu@unity.rc.umass.edu:/work/toltec/commissioning2025-C1/2025-C1-COM-14/maxm/science/reduce_rho_oph_eval/redu<redu_number> .", where redu_number is the number of the reduction folder you want. If the reduction has not yet been aligned, run align_observation.ipynb with redu_num set to the reduction number. If the reduction has been corrected, run correct_position.ipynb with redu_num set to the reduction number. 

## align_observations.ipynb
This notebook finds the local Alt/Az offsets necessary to align individual observations into a coherent final coadded image. These offsets are found via Enhanced Cross Correlation Maximization, which figures out the best x-y offsets with which to align each observation to a reference observation. These offsets should be inputted into the associated .yaml file in Tolteca under pointing_offsets.

## correct_position.ipynb
This notebook makes a necessary positioning correction to the final coadded image. This is done by choosing a point source (VLA 1623-243 in Rho Oph A is given as an example), and comparing its actual position to what the image thinks its position is. The source's position in the image is found via iterative centroiding.  The corrected image will have a '_centered' added to it and be located in the same folder as the uncorrected coaddded image

## offset_functions.py
This python file contains the two most vital functions in the above notebooks: relative_pointing_offsets and absolute_pointing_offset. They should be looked at and scrutinized if any errors occur.
