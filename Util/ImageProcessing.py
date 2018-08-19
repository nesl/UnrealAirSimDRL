import scipy as sci
import numpy as np
import ImageKernals as IK


# Gather SVDs, Guassian Blur, Horiz Edge, Vert Edge Imgs in Dic
def get_image_transforms(images):
    # Gaussian Kernal
    gk = IK.get_gaussian_kernal((5,5), intensity = .05)
    
    # Apply Edge Detection:
    khorz = IK.get_horz_edge_kern()
    kvert = IK.get_vert_edge_kern()
    
    # Returns a dictionary of your images transformed, per enteries of the dictionary
    img_dic = {"Raw":images,
               "Gaus":[sci.signal.convolve2d(images[i],gk, mode = 'same') for i in range(images.shape[0])], 
               "SVD": [np.linalg.svd(images[i]) for i in range(images.shape[0])], 
               "VED":[sci.signal.convolve2d(images[i],kvert,mode = 'same') for i in range(images.shape[0])], 
               "HED":[sci.signal.convolve2d(images[i],khorz,mode = 'same') for i in range(images.shape[0])], }
    return img_dic

# Utility for appendning multiple vectors, images, matrices, or 4-d tensors,
    # where obs4 is the stacked element and obs is the new chunck of data to stack in
def trim_append_state_vector(obs4, obs, repeat = 1, pop_index = 1): 
        # assumed obs4 and obs have the same nominal dimension
        print(obs.shape, obs4.shape)
        for i in range(repeat):
            obs_shape = obs.shape
            if len(obs_shape) == 2:
                obs = obs.reshape(obs_shape[0], obs_shape[1], 1)
                obs4 = np.dstack((obs4[:,:,1:], obs))
            elif len(obs_shape) == 3:
                obs4 = np.dstack((obs4[:,:,pop_index:], obs))
        return obs4


# Assure you are good for v_stack here
def fill_state_vector(obs3, repeat = 2): 
        # assumed obs4 and obs have the same nominal dimension
        obs4 = None
        obs3_shape = obs3.shape
        
        if len(obs3.shape) == 2:
            obs3 = obs3.reshape(obs3_shape[0], obs3_shape[1], 1)
            obs4 = obs3
            for i in range(repeat):
                    obs4 = np.dstack((obs4,obs3))
        elif len(obs3.shape) == 3:
            obs4 = obs3
            for i in range(repeat):
                    obs4 = np.dstack((obs4,obs3))








