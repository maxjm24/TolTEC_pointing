import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2

# Finds the offsets between ref_obs and each image in align_obsvs that maximize the cross-correlation between them
def relative_pointing_offsets(ref_obs, ref_center, align_obsvs, align_centers, wav, redu_num, crop, num_iter, min_increment, x_push, y_push):
    
    # The limits of the reference image are defined by a square of sidelength crop centered on the image's centerpixel and offset by x_push/y_push
    ref_y_min, ref_y_max = ref_center[0]-crop-1+y_push, ref_center[0]+crop+y_push
    ref_x_min, ref_x_max = ref_center[1]-crop-1+x_push, ref_center[1]+crop+x_push
    # Obtains the reference observation image within the limits defined above at total intensity, assuming the below filepath
    ref_img = fits.open('./redu' + redu_num + '/' + ref_obs + '/raw/toltec_commissioning_a' + wav + '_science_' 
                            + ref_obs + '_citlali.fits')[1].data[0][0][ref_y_min:ref_y_max, ref_x_min:ref_x_max].astype(np.float32)
    print(f'Reference Observation: {ref_obs}')
    print()
    
    # Motion model (MOTION_TRANSLATION is just x-y translation, MOTION_EUCLIDEAN is translation and rotation, etc.)
    motion_type = cv2.MOTION_TRANSLATION
    
    # 2x3 transformation matrix for a 2D transformation
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Terminates after num_iter iterations or when the cross-correlation changes by less than min_increment
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, min_increment)
    size = ref_img.shape
    
    align_imgs, aligned_imgs, x_shifts, y_shifts = [], [], [], [] # List of unaligned images, list of aligned images, and offset lists
    for (align_obs, align_center) in zip(align_obsvs, align_centers): # Loops over each observation image to align
        
        # The limits of the image are defined by a square of sidelength crop centered on the image's centerpixel and offset by x_push/y_push
        align_y_min, align_y_max = align_center[0]-crop-1+y_push, align_center[0]+crop+y_push
        align_x_min, align_x_max = align_center[1]-crop-1+x_push, align_center[1]+crop+x_push
        # Obtains the observation image within the limits defined above at total intensity, assuming the below filepath
        align_img = fits.open('./redu' + redu_num + '/' + align_obs + '/raw/toltec_commissioning_a' + wav + '_science_' 
                              + align_obs + '_citlali.fits')[1].data[0][0][align_y_min:align_y_max, align_x_min:align_x_max].astype(np.float32)
        align_imgs.append(align_img)
        
        # Runs the Enhanced Cross Correlation (ECC) Maximation algorithm between ref_img and align_img, with x-y translation and specified criteria
        cc, warp_matrix = cv2.findTransformECC(ref_img, align_img, warp_matrix, motion_type, criteria)
    
        # The maximized transformation matrix is applied to align the image to ref_img
        aligned_img = cv2.warpAffine(align_img, warp_matrix, (size[1], size[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned_imgs.append(aligned_img)

        # x/y pixel shifts necessary to align the image to ref_img
        x_shift = -warp_matrix[0, 2]
        y_shift = -warp_matrix[1, 2]
        x_shifts.append(x_shift)
        y_shifts.append(y_shift)
        
        print(f'Observation {align_obs}')
        print(f'Maximized Enhanced Correlation Coefficient: {cc}')
        print(f'X-shift (Arcseconds): {x_shift}')
        print(f'Y-shift (Arcseconds): {y_shift}')
        print()
    
    align_imgs = np.array(align_imgs)
    aligned_imgs = np.array(aligned_imgs)
    comp_img = (ref_img + align_imgs.sum(axis=0))/(len(align_obsvs) + 1) # Average of all images (composite image) before alignment
    aligned_comp_img = (ref_img + aligned_imgs.sum(axis=0))/(len(align_obsvs) + 1) # Composite image after alignment

    # Plots the composite images before/after alignment to show what was improved
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(8)
    fig.set_figwidth(10)
    im = ax[0].imshow(comp_img, cmap='bone', origin='lower')
    ax[0].set_title(f'Unaligned Composite Image ({wav} micron)')
    im = ax[1].imshow(aligned_comp_img, cmap='bone', origin='lower')
    ax[1].set_title(f'Aligned Composite Image ({wav} micron)')
    plt.subplots_adjust(wspace = 0.3)
    fig.colorbar(im, ax=ax, cax=fig.add_axes([0.95, 0.3, 0.03, 0.4]));

    # returns the x/y pixel shifts necessary to align each image to ref_img, along with the aligned composite image
    return x_shifts, y_shifts, aligned_comp_img