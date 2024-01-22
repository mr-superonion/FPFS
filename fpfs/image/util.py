import jax
import jax.numpy as jnp


def get_klim(psf_array, sigma, thres=1e-20):
    """Gets klim, the region outside klim is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros

    Args:
    psf_array (ndarray):    PSF's Fourier power or Fourier transform
    sigma (float):          one sigma of Gaussian Fourier power
    thres (float):          the threshold for a tuncation on Gaussian
                                [default: 1e-20]
    Returns:
    klim (float):           the limit radius
    """
    ngrid = psf_array.shape[0]

    def cond_fun(dist):
        v1 = abs(
            jnp.exp(-(dist**2.0) / 2.0 / sigma**2.0)
            / psf_array[ngrid // 2 + dist, ngrid // 2]
        )
        v2 = abs(
            jnp.exp(-(dist**2.0) / 2.0 / sigma**2.0)
            / psf_array[ngrid // 2, ngrid // 2 + dist]
        )
        return jax.lax.cond(
            v1 < v2,
            v1,
            lambda x: x > thres,
            v2,
            lambda x: x > thres,
        )

    def body_fun(dist):
        return dist + 1

    klim = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=ngrid // 5,
    )
    return klim


def gauss_kernel_fft(ny, nx, sigma, klim, return_grid=False):
    """Generates a Gaussian kernel on grids for np.fft.fft transform
    (we always shift k=0 to (ngird//2, ngird//2)). The kernel is truncated at
    radius klim.

    Args:
    ny (int):    		    grid size in y-direction
    nx (int):    		    grid size in x-direction
    sigma (float):		    scale of Gaussian in Fourier space
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


def gauss_kernel_rfft(ny, nx, sigma, klim, return_grid=False):
    """Generates a Gaussian kernel on grids for np.fft.rfft transform
    The kernel is truncated at radius klim.

    Args:
    ny (int):    		    grid size in y-direction
    nx (int):    		    grid size in x-direction
    sigma (float):		    scale of Gaussian in Fourier space
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
    mask = (r2 <= klim**2).astype(jnp.float64)
    out = jnp.exp(-r2 / 2.0 / sigma**2.0) * mask
    if not return_grid:
        return out
    else:
        return out, (ygrid, xgrid)


def get_fourier_pow_fft(input_data):
    """Gets Fourier power function

    Args:
    input_data (ndarray):  image array, centroid does not matter.

    Returns:
    out (ndarray):      Fourier Power
    """
    out = (jnp.abs(jnp.fft.fft2(input_data)) ** 2.0).astype(jnp.float64)
    out = jnp.fft.fftshift(out)
    return out


def get_fourier_pow_rfft(input_data):
    """Gets Fourier power function

    Args:
    input_data (ndarray):  image array. The centroid does not matter.

    Returns:
    galpow (ndarray):   Fourier Power
    """

    out = (jnp.abs(jnp.fft.rfft2(input_data)) ** 2.0).astype(jnp.float64)
    return out
