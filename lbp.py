import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA

def lbp_descriptor(image, cropX=100, cropY=170):
    """
    Computes the LBP descriptor normalized to unit length for a given image path

    Crops the centre (cropX, cropY) region of source image to calculate LBP
    descriptor

    Parameters
    ----------
    img_path: str
        Path to image.
    cropX: int
        Number of pixels to crop from original image horizontally
    cropY: int
        Number of pixels to crop from original image vertically

    Returns
    -------
    desc: ndarray
        LBP descriptor of source image
    """
    descriptors = []
    tensor = image.cpu()
    for imge in tensor:
        
        img = imge.numpy()
        start_cropY = int((img.shape[0] - cropY) / 2)
        start_cropX = int((img.shape[1] - cropX) / 2)

        cropped = img[
            start_cropY : start_cropY + cropY, start_cropX : start_cropX + cropX
        ]

        # LBP params
        radius = 1
        n_points = 8 * radius
        cell_size = (8, 8)
        METHOD = "uniform"

        desc = []
        for y in range(0, cropped.shape[0], cell_size[0]):
            for x in range(0, cropped.shape[1], cell_size[1]):
                cell = cropped[y : y + cell_size[0], x : x + cell_size[1]]
                lbp = local_binary_pattern(cell, n_points, radius, METHOD)
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=np.arange(0, n_points + 3),
                    range=(0, n_points + 2),
                )

                desc.append(hist)

        # desc = np.concatenate(desc)

        # Normalize vector to unit length
        # desc = desc / np.linalg.norm(desc)
        descriptors.append(desc)
    return np.array(descriptors)

def pca_filter(desc, sampling_stride=5, cropX=100, cropY=170, cell_size=(8,8), n_dimensions=5):
    """
    Applies PCA on the lbp descriptor

    Parameters
    ----------
    sampling_stride: int
        Parameter of the pca filter
    cropX: int
        Number of pixels to crop from original image horizontally
    cropY: int
        Number of pixels to crop from original image vertically
    
    Returns
    -------
    desc: ndarray
        LBP descriptor of source image after applying pca filter
    """
    num_rows = int(cropY /cell_size[1])
    num_cols = int(cropX / cell_size[0])
    pca_desc = []

    for u in range(0, num_rows, sampling_stride):
        for v in range(0, num_cols, sampling_stride):
            pca_desc.append(desc[u*num_cols+v])

    pca = PCA(n_components=n_dimensions)
    new_desc = pca.fit_transform(np.array(pca_desc))
    new_desc = np.concatenate(new_desc)
    
    return new_desc