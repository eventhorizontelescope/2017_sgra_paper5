def convolveSquareImage(image, fieldOfView, gaussian_fwhm, nan_treatment='interpolate'):
    from astropy.convolution import convolve_fft, Gaussian2DKernel
    import numpy as np

    assert image.shape[0] == image.shape[1]

    smoothingIndices = gaussian_fwhm / 2.355 / (fieldOfView / image.shape[0])
    kernel = Gaussian2DKernel(smoothingIndices)
    output = convolve_fft(image, kernel, nan_treatment=nan_treatment)
    return output

