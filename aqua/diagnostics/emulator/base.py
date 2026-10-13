"""Shared infrastructure for diagnostics comparing emulator and reference data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import DEFAULT_REALIZATION
from aqua.diagnostics.base import Diagnostic, OutputSaver


class EmulatorDiagnostic(Diagnostic):
    """Base class for diagnostics operating on a prediction/reference pair.

    The primary dataset arguments are the standard :class:`Diagnostic`
    arguments.  Reference metadata uses the ``ref_*`` arguments.  Both
    datasets are retrieved through :meth:`Diagnostic._retrieve`, so AQUA's
    normal selection, fixer and regridding behaviour is preserved.
    """

    def __init__(
        self,
        *args,
        ref_model=None,
        ref_exp=None,
        ref_source=None,
        ref_catalog=None,
        ref_regrid=None,
        diagnostic_name="emulator",
        outputdir=".",
        rebuild=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = log_configure(
            log_name=self.__class__.__name__, log_level=self.loglevel
        )
        self.ref_model = ref_model
        self.ref_exp = ref_exp
        self.ref_source = ref_source
        self.ref_catalog = ref_catalog
        self.ref_regrid = ref_regrid
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.data_ref = None
        self.reader = None
        self.reader_ref = None
        self.realization_ref = None
        self.outputsaver = OutputSaver(
            diagnostic=diagnostic_name,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            realization=self.realization,
            catalog_ref=ref_catalog,
            model_ref=ref_model,
            exp_ref=ref_exp,
            outputdir=outputdir,
            loglevel=self.loglevel,
        )

    @staticmethod
    def _select_level(data: xr.Dataset | xr.DataArray, level: int | float | None):
        """Select and drop a requested pressure level when present."""
        if level is None:
            return data
        for coord in ("plev", "level", "lev", "pressure"):
            if coord not in data.coords and coord not in data.dims:
                continue
            if data.sizes.get(coord, 1) == 1:
                return data.isel({coord: 0}, drop=True)
            try:
                return data.sel({coord: level}, drop=True)
            except (KeyError, ValueError):
                return data.sel({coord: level}, method="nearest", drop=True)
        return data

    def retrieve_pair(
        self,
        var,
        level=None,
        reader_kwargs=None,
        ref_reader_kwargs=None,
    ):
        """Retrieve one prediction/reference variable pair.

        Regridding is performed by :meth:`Diagnostic._retrieve`.  If a level
        was requested, the resulting singleton pressure coordinate is dropped.
        The aligned datasets and readers are retained on the instance.
        """
        reader_kwargs = dict(reader_kwargs or {})
        ref_reader_kwargs = dict(ref_reader_kwargs or {})

        self.data, self.reader, self.catalog = self._retrieve(
            model=self.model,
            exp=self.exp,
            source=self.source,
            var=var,
            catalog=self.catalog,
            startdate=self.startdate,
            enddate=self.enddate,
            regrid=self.regrid,
            reader_kwargs=reader_kwargs,
            loglevel=self.loglevel,
        )
        self.data_ref, self.reader_ref, self.ref_catalog = self._retrieve(
            model=self.ref_model,
            exp=self.ref_exp,
            source=self.ref_source,
            var=var,
            catalog=self.ref_catalog,
            startdate=self.startdate,
            enddate=self.enddate,
            regrid=self.ref_regrid,
            reader_kwargs=ref_reader_kwargs,
            loglevel=self.loglevel,
        )

        self.data = self._select_level(self.data, level)
        self.data_ref = self._select_level(self.data_ref, level)
        self.data, self.data_ref = self.align(self.data, self.data_ref)
        self.realization = reader_kwargs.get("realization", DEFAULT_REALIZATION)
        self.realization_ref = ref_reader_kwargs.get(
            "realization", DEFAULT_REALIZATION
        )

        # Catalogs may be inferred by Reader, so refresh the saver after retrieval.
        self.outputsaver = self._make_outputsaver()
        return self.data, self.data_ref

    @staticmethod
    def iter_variables(variables):
        """Yield ``(name, level, key)`` with normalized level specifications.

        ``variables`` accepts names, variable dictionaries, or a single
        dictionary.  Missing levels produce the variable name as key; pressure
        levels produce keys such as ``zg_50000``.
        """
        if variables is None:
            return
        if isinstance(variables, (str, dict)):
            variables = [variables]
        if not isinstance(variables, Iterable):
            raise TypeError("variables must be a variable or an iterable")

        for var_info in variables:
            if isinstance(var_info, str):
                name, levels = var_info, None
            elif isinstance(var_info, dict):
                name = var_info.get("name")
                levels = var_info.get("level")
            else:
                raise TypeError("each variable must be a name or mapping")
            if not name:
                raise ValueError("variable entries require a non-empty 'name'")

            if levels is None:
                level_values = [None]
            elif np.isscalar(levels):
                level_values = [levels]
            else:
                level_values = list(levels)

            for level in level_values:
                normalized = None if level is None else int(level)
                key = name if normalized is None else f"{name}_{normalized}"
                yield name, normalized, key

    @staticmethod
    def align(pred, ref):
        """Align time and harmonize latitude/longitude coordinate dtypes.

        Latitude and longitude coordinates are converted to ``float32`` in both
        inputs before an inner alignment.  The latter guarantees that only the
        common analysis period is retained.
        """
        pred = pred.copy()
        ref = ref.copy()
        for data_name, data in (("prediction", pred), ("reference", ref)):
            updates = {}
            for coord in data.coords:
                lowered = coord.lower()
                if "lat" in lowered or "lon" in lowered:
                    if np.issubdtype(data[coord].dtype, np.floating):
                        updates[coord] = data[coord].astype(np.float32)
            if updates:
                data = data.assign_coords(updates)
            if data_name == "prediction":
                pred = data
            else:
                ref = data

        if "time" in pred.coords and "time" in ref.coords:
            common_time = np.intersect1d(pred.time.values, ref.time.values)
            pred = pred.sel(time=common_time)
            ref = ref.sel(time=common_time)
        return xr.align(pred, ref, join="inner")

    def _make_outputsaver(self):
        return OutputSaver(
            diagnostic=self.diagnostic_name,
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            realization=self.realization,
            catalog_ref=self.ref_catalog,
            model_ref=self.ref_model,
            exp_ref=self.ref_exp,
            outputdir=self.outputdir,
            loglevel=self.loglevel,
        )

    def save_pair_netcdf(
        self, data, diagnostic_product, extra_keys=None, **kwargs
    ):
        """Save paired diagnostic output with prediction/reference metadata."""
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise TypeError("data must be an xarray Dataset or DataArray")
        if isinstance(data, xr.DataArray):
            data = data.to_dataset(name=data.name or diagnostic_product)
        rebuild = kwargs.pop("rebuild", self.rebuild)
        return self.outputsaver.save_netcdf(
            dataset=data,
            diagnostic_product=diagnostic_product,
            rebuild=rebuild,
            extra_keys=extra_keys,
            **kwargs,
        )


class EmulatorPlotMixin:
    """OutputSaver-backed figure persistence for emulator plot classes."""

    def __init__(
        self,
        *,
        diagnostic_name="emulator",
        catalog=None,
        model=None,
        exp=None,
        realization=None,
        ref_catalog=None,
        ref_model=None,
        ref_exp=None,
        outputdir=".",
        rebuild=True,
        loglevel="WARNING",
        **_,
    ):
        self.diagnostic_name = diagnostic_name
        self.outputdir = outputdir
        self.rebuild = rebuild
        self.loglevel = loglevel
        self.logger = log_configure(
            log_name=self.__class__.__name__, log_level=loglevel
        )
        self.outputsaver = OutputSaver(
            diagnostic=diagnostic_name,
            catalog=catalog,
            model=model,
            exp=exp,
            realization=realization,
            catalog_ref=ref_catalog,
            model_ref=ref_model,
            exp_ref=ref_exp,
            outputdir=outputdir,
            loglevel=loglevel,
        )

    @classmethod
    def from_diagnostic(cls, diagnostic: EmulatorDiagnostic, **kwargs):
        """Build a plotter using metadata from a computation object."""
        return cls(
            diagnostic_name=diagnostic.diagnostic_name,
            catalog=diagnostic.catalog,
            model=diagnostic.model,
            exp=diagnostic.exp,
            realization=diagnostic.realization,
            ref_catalog=diagnostic.ref_catalog,
            ref_model=diagnostic.ref_model,
            ref_exp=diagnostic.ref_exp,
            outputdir=diagnostic.outputdir,
            rebuild=diagnostic.rebuild,
            loglevel=diagnostic.loglevel,
            **kwargs,
        )

    def save_plot(
        self,
        fig,
        diagnostic_product,
        extra_keys=None,
        metadata=None,
        dpi=300,
        format="pdf",
    ):
        """Save a figure and attach standard pair metadata."""
        return self.outputsaver.save_figure(
            fig=fig,
            diagnostic_product=diagnostic_product,
            extra_keys=extra_keys,
            metadata=metadata,
            extension=format,
            rebuild=self.rebuild,
            dpi=dpi,
        )
