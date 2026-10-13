"""Anomaly Correlation Coefficient diagnostic."""

from __future__ import annotations

import xarray as xr

from ..base import EmulatorDiagnostic
from ..util import as_dataarray, horizontal_dims


class ACC(EmulatorDiagnostic):
    """Compute temporal and spatial anomaly correlation coefficients."""

    def __init__(
        self,
        *args,
        variables=None,
        clim_model=None,
        clim_exp=None,
        clim_source=None,
        clim_catalog=None,
        clim_regrid=None,
        clim_startdate=None,
        clim_enddate=None,
        **kwargs,
    ):
        kwargs.setdefault("diagnostic_name", "acc")
        super().__init__(*args, **kwargs)
        self.variables = variables or []
        self.clim_model = clim_model
        self.clim_exp = clim_exp
        self.clim_source = clim_source
        self.clim_catalog = clim_catalog
        self.clim_regrid = clim_regrid
        self.clim_startdate = clim_startdate
        self.clim_enddate = clim_enddate
        self.spatial_acc = {}
        self.temporal_acc = {}

    @staticmethod
    def anomalies(data, climatology_mean):
        """Subtract a configured external climatology mean."""
        result = data - climatology_mean
        result.attrs = data.attrs
        return result

    @staticmethod
    def spatial(pred, ref, climatology_mean):
        """Correlation over time at every grid point."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        clim = climatology_mean
        return xr.corr(
            ACC.anomalies(pred, clim),
            ACC.anomalies(ref, clim),
            dim="time",
        ).rename("acc")

    @staticmethod
    def temporal(pred, ref, climatology_mean):
        """Spatial pattern correlation at every time step."""
        pred, ref = EmulatorDiagnostic.align(pred, ref)
        dims = horizontal_dims(pred)
        if not dims:
            raise ValueError("unable to determine horizontal dimensions for ACC")
        return xr.corr(
            ACC.anomalies(pred, climatology_mean),
            ACC.anomalies(ref, climatology_mean),
            dim=dims,
        ).rename("acc")

    def retrieve_climatology(self, var, level=None, reader_kwargs=None):
        """Retrieve climatology from the source supplied in configuration."""
        required = (self.clim_model, self.clim_exp, self.clim_source)
        if any(item is None for item in required):
            raise ValueError(
                "ACC requires configured clim_model, clim_exp and clim_source"
            )
        data, reader, self.clim_catalog = self._retrieve(
            model=self.clim_model,
            exp=self.clim_exp,
            source=self.clim_source,
            var=var,
            catalog=self.clim_catalog,
            startdate=self.clim_startdate,
            enddate=self.clim_enddate,
            regrid=self.clim_regrid,
            reader_kwargs=dict(reader_kwargs or {}),
            loglevel=self.loglevel,
        )
        data = self._select_level(data, level)
        da = as_dataarray(data, var)
        return reader.timmean(da) if "time" in da.dims else da

    def evaluate(self, pred, ref, climatology_mean, *, key="variable"):
        """Compute and retain both ACC products."""
        spatial = self.spatial(pred, ref, climatology_mean)
        temporal = self.temporal(pred, ref, climatology_mean)
        self.spatial_acc[key] = spatial
        self.temporal_acc[key] = temporal
        return spatial, temporal

    def run(
        self,
        variables=None,
        *,
        climatologies=None,
        reader_kwargs=None,
        ref_reader_kwargs=None,
        clim_reader_kwargs=None,
        save_netcdf=False,
    ):
        """Compute ACC using configured or explicitly supplied climatologies."""
        climatologies = climatologies or {}
        for var, level, key in self.iter_variables(variables or self.variables):
            pred, ref = self.retrieve_pair(
                var, level, reader_kwargs, ref_reader_kwargs
            )
            climatology = climatologies.get(key)
            if climatology is None:
                climatology = self.retrieve_climatology(
                    var, level, clim_reader_kwargs
                )
            spatial, temporal = self.evaluate(
                as_dataarray(pred, var),
                as_dataarray(ref, var),
                as_dataarray(climatology, var),
                key=key,
            )
            if save_netcdf:
                extra = {"var": var, "level": level}
                self.save_pair_netcdf(
                    spatial.to_dataset(), "spatial", extra_keys=extra
                )
                self.save_pair_netcdf(
                    temporal.to_dataset(), "temporal", extra_keys=extra
                )
        return {"spatial": self.spatial_acc, "temporal": self.temporal_acc}
