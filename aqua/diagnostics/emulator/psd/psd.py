"""Power spectral density comparison for regular and HEALPix grids."""

from __future__ import annotations

import numpy as np
import numpy.fft as fft
import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray

try:
    import healpy as hp
except ImportError:  # pragma: no cover - optional dependency
    hp = None


class PSD(EmulatorDiagnostic):
    """Compute FFT spectra on lat/lon grids or ``healpy.anafast`` on HEALPix."""

    def __init__(
        self,
        *args,
        variables=None,
        time_averaged_psd=True,
        psd_of_time_mean=True,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "psd")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.do_time_averaged_psd = time_averaged_psd
        self.do_psd_of_time_mean = psd_of_time_mean
        self.results = {}

    @staticmethod
    def radial_average(power_2d):
        """Calculate the frontier radial average of a 2-D power spectrum."""
        y_center, x_center = np.array(power_2d.shape) // 2
        y, x = np.indices(power_2d.shape)
        radius = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2).astype(
            np.int32
        )
        total = np.bincount(radius.ravel(), power_2d.ravel())
        count = np.bincount(radius.ravel())
        return np.divide(
            total,
            count,
            out=np.zeros_like(total, dtype=float),
            where=count != 0,
        )

    _radial_average = radial_average

    @staticmethod
    def _healpix_dim(data):
        for dim in ("ncells", "cell"):
            if dim in data.dims:
                return dim
        return None

    @classmethod
    def is_healpix(cls, data):
        dim = cls._healpix_dim(data)
        if dim and (hp is None or hp.isnpixok(data.sizes[dim])):
            return True
        attrs = " ".join(
            str(data.attrs.get(key, "")) for key in ("grid", "AQUA_grid")
        ).lower()
        return "healpix" in attrs

    @staticmethod
    def _nested(data):
        values = [
            data.attrs.get(key)
            for key in ("healpix_order", "healpix_ordering", "ordering", "order")
        ]
        return any("nest" in str(value).lower() for value in values if value)

    @classmethod
    def fft_spectrum(cls, data, *, average_time=True):
        """Apply the unchanged frontier FFT/radial-average numerical core."""
        values = np.nan_to_num(np.asarray(data), nan=0.0)
        if average_time:
            if values.ndim == 2:
                values = values[np.newaxis, ...]
            transformed = fft.fft2(values, axes=(-2, -1))
            power = np.abs(fft.fftshift(transformed, axes=(-2, -1))) ** 2
            return np.mean([cls.radial_average(frame) for frame in power], axis=0)
        transformed = fft.fftshift(fft.fft2(values))
        return cls.radial_average(np.abs(transformed) ** 2)

    @classmethod
    def healpix_spectrum(cls, data, *, average_time=True):
        """Apply ``healpy.anafast`` after conversion to ring ordering."""
        if hp is None:
            raise ImportError("healpy is required to process HEALPix grids")
        cell = cls._healpix_dim(data)
        if cell is None:
            raise ValueError("could not identify the HEALPix cell dimension")
        order = [dim for dim in data.dims if dim != cell] + [cell]
        maps = np.nan_to_num(data.transpose(*order).values, nan=0.0)
        maps = maps.reshape(-1, maps.shape[-1])
        if cls._nested(data):
            maps = np.asarray([hp.reorder(field, n2r=True) for field in maps])
        nside = hp.npix2nside(maps.shape[-1])
        spectra = [hp.anafast(field, lmax=3 * nside - 1) for field in maps]
        return np.mean(spectra, axis=0) if average_time else spectra[0]

    @staticmethod
    def _dataset(pred, ref):
        length = max(len(pred), len(ref))
        return xr.Dataset(
            {
                "psd_prediction": (
                    "wavenumber",
                    np.pad(pred, (0, length - len(pred)), constant_values=np.nan),
                ),
                "psd_reference": (
                    "wavenumber",
                    np.pad(ref, (0, length - len(ref)), constant_values=np.nan),
                ),
            },
            coords={"wavenumber": np.arange(length)},
        )

    def evaluate(self, pred, ref, *, key="variable"):
        """Compute selected PSD products for one aligned pair."""
        pred, ref = self.align(pred, ref)
        healpix = self.is_healpix(pred) and self.is_healpix(ref)
        method = self.healpix_spectrum if healpix else self.fft_spectrum
        result = {}
        if self.do_time_averaged_psd:
            result["time_averaged_psd"] = self._dataset(
                method(pred, average_time=True),
                method(ref, average_time=True),
            )
        if self.do_psd_of_time_mean:
            pred_mean = pred.mean("time") if "time" in pred.dims else pred
            ref_mean = ref.mean("time") if "time" in ref.dims else ref
            result["psd_of_time_mean"] = self._dataset(
                method(pred_mean, average_time=False),
                method(ref_mean, average_time=False),
            )
        for dataset in result.values():
            dataset.attrs["spectral_method"] = (
                "healpy.anafast" if healpix else "2-D FFT radial average"
            )
        self.results[key] = result
        return result

    def run(
        self,
        variables=None,
        *,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        save_netcdf=False,
    ):
        for var, level, key in self.iter_variables(variables or self.variables):
            pred, ref = self.retrieve_pair(
                var, level, reader_kwargs, ref_reader_kwargs
            )
            result = self.evaluate(
                as_dataarray(pred, var), as_dataarray(ref, var), key=key
            )
            if save_netcdf:
                for product, dataset in result.items():
                    self.save_pair_netcdf(
                        dataset,
                        product,
                        extra_keys={"var": var, "level": level},
                    )
        return self.results

    calculate_and_plot_psd = run
