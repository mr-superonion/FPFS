import math

import jax.numpy as jnp


def get_klim(psf_pow, sigma: float, thres: float = 1e-20) -> float:
    """Gets klim, the region outside klim is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros

    Args:
    psf_pow (ndarray):      PSF's Fourier power (rfft)
    sigma (float):          one sigma of Gaussian Fourier power (pixel scale=1)
    thres (float):          the threshold for a tuncation on Gaussian
                                [default: 1e-20]
    Returns:
    klim (float):           the limit radius
    """
    ngrid = psf_pow.shape[0]
    gaussian, (y, x) = gauss_kernel_rfft(ngrid, ngrid, sigma, jnp.pi, return_grid=True)
    r = jnp.sqrt(x**2.0 + y**2.0)  # radius
    mask = gaussian / psf_pow < thres
    dk = 2.0 * math.pi / ngrid
    klim_pix = round(float(jnp.min(r[mask]) / dk))
    klim_pix = min(max(klim_pix, ngrid // 5), ngrid // 2 - 1)
    return klim_pix


def gauss_kernel_fft(
    ny: int, nx: int, sigma: float, klim: float, return_grid: bool = False
):
    """Generates a Gaussian kernel on grids for np.fft.fft transform
    (we always shift k=0 to (ngird//2, ngird//2)). The kernel is truncated at
    radius klim.

    Args:
    ny (int):    		    grid size in y-direction
    nx (int):    		    grid size in x-direction
    sigma (float):		    scale of Gaussian in Fourier space (pixel scale=1)
    klim (float):           upper limit of k
    return_grid (bool):     return grids [True] or not [Flase]
                                [default: False]
    Returns:
    out (ndarray):          Gaussian on grids
    xgrid,ygrid (typle):    grids for [y, x] axes if return_grid
    """
    # mask
    x = jnp.fft.fftshift(jnp.fft.fftfreq(nx, 1 / jnp.pi / 2.0))
    y = jnp.fft.fftshift(jnp.fft.fftfreq(ny, 1 / jnp.pi / 2.0))
    ygrid, xgrid = jnp.meshgrid(y, x, indexing="ij")
    r2 = xgrid**2.0 + ygrid**2.0
    mask = (r2 <= klim**2).astype(jnp.float64)
    out = jnp.exp(-r2 / 2.0 / sigma**2.0) * mask
    if not return_grid:
        return out
    else:
        return out, (ygrid, xgrid)


def gauss_kernel_rfft(
    ny: int, nx: int, sigma: float, klim: float, return_grid: bool = False
):
    """Generates a Gaussian kernel on grids for np.fft.rfft transform
    The kernel is truncated at radius klim.

    Args:
    ny (int):    		    grid size in y-direction
    nx (int):    		    grid size in x-direction
    sigma (float):		    scale of Gaussian in Fourier space (pixel scale=1)
    klim (float):           upper limit of k
    return_grid (bool):     return grids or not

    Returns:
    out (ndarray):          Gaussian on grids
    ygrid, xgrid (typle):   grids for [y, x] axes, if return_grid
    """
    x = jnp.fft.rfftfreq(nx, 1 / jnp.pi / 2.0)
    y = jnp.fft.fftfreq(ny, 1 / jnp.pi / 2.0)
    ygrid, xgrid = jnp.meshgrid(y, x, indexing="ij")
    r2 = xgrid**2.0 + ygrid**2.0
    mask = (r2 <= klim**2).astype(int)
    out = jnp.exp(-r2 / 2.0 / sigma**2.0) * mask
    if not return_grid:
        return out
    else:
        return out, (ygrid, xgrid)


def truncate_square(arr, rcut: int) -> None:
    """Truncate the input array with square

    Args:
    arr (ndarray):      image array
    rcut (int):         radius of the square (width / 2)
    """
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")

    ngrid = arr.shape[0]
    arr[: ngrid // 2 - rcut, :] = 0
    arr[ngrid // 2 + rcut :, :] = 0
    arr[:, : ngrid // 2 - rcut] = 0
    arr[:, ngrid // 2 + rcut :] = 0
    return


def truncate_circle(arr, rcut: float) -> None:
    """Truncate the input array with circle

    Args:
    arr (ndarray):      image array
    rcut (float):       radius of the circle (width / 2)
    """
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")
    ngrid = arr.shape[0]
    y, x = jnp.ogrid[0:ngrid, 0:ngrid]
    center_x, center_y = ngrid // 2, ngrid // 2
    # Compute the squared distance to the center
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    # Mask values outside the circle
    arr[distance_squared > rcut**2] = 0.0
    return


def truncate_psf_fft(arr, rcut: float):
    """Truncate the input PSF array in Fourier space with circle

    Args:
    arr (ndarray):      image array
    rcut (float):       radius of the circle (width / 2)

    Returns:
    out (ndarray):      truncated array
    """
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")
    ngrid = arr.shape[0]
    center = ngrid // 2
    # Create a meshgrid for x and y coordinates
    y, x = jnp.ogrid[:ngrid, :ngrid]
    # Calculate the distance from the center
    distance_squared = (x - center) ** 2 + (y - center) ** 2.0
    # Create a mask for pixels outside the radius rcut
    mask = distance_squared > rcut**2.0
    # Set pixels outside rcut to 1e5
    out = jnp.where(mask, 1e5, arr)
    return out


def truncate_psf_rfft(arr, klim_pix: float, ngrid: int):
    """Truncate the input PSF array in Fourier (rfft) space with circle.
    Note that this function is only used for truncation in PSF deconvolution,
    and the input array is the psf to be deconvolved

    Args:
    arr (ndarray):      image array
    klim_pix (float):   radius of the circle in units of pixel
    ngrid (int):        number of pixel grids

    Returns:
    out (ndarray):      truncated array
    """
    if len(arr.shape) != 2 or arr.shape != (ngrid, ngrid // 2 + 1):
        raise ValueError("Input array must be a 2D square array")
    # Create a meshgrid of frequency values
    fy, fx = jnp.meshgrid(
        jnp.fft.fftfreq(ngrid) * ngrid,
        jnp.fft.rfftfreq(ngrid) * ngrid,
        indexing="ij",
    )

    # Calculate distances in the frequency domain
    d2 = fx**2 + fy**2
    # Create a mask for distances greater tha  klim_pix
    mask = d2 > klim_pix**2.0
    # Apply the mask
    out = jnp.where(mask, 1e5, arr)
    return out


def rotate90(image):
    rotated_image = jnp.zeros_like(image)
    rotated_image = rotated_image.at[1:, 1:].set(jnp.rot90(m=image[1:, 1:], k=-1))
    return rotated_image
