import numpy as np
import cv2

def get_gaussian_psf_kernel(sigma: float, size: int = 7) -> np.ndarray:
    """Generate a 2D Gaussian kernel.
    
    Args:
        sigma: Standard deviation of the Gaussian kernel.
        size: Size of the kernel (width and height).
        
    Returns:
        2D numpy array with the Gaussian kernel, normalized to sum to 1.
    """
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def get_psf_kernels_dict(sigma: float, bands: list[str], size: int = 7) -> dict[str, np.ndarray]:
    """Generate a dictionary of Gaussian kernels for each band.
    
    Args:
        sigma: Standard deviation of the Gaussian kernel.
        bands: List of band names.
        size: Size of the kernel (width and height).
        
    Returns:
        Dictionary mapping band names to 2D numpy arrays.
    """
    kernel = get_gaussian_psf_kernel(sigma, size)
    return {band: kernel for band in bands}
