from ophyd import Device, Component as Cpt
from ophyd.sim import SynAxis, SynSignalRO, SynSignal
import numpy as np
import scipy.interpolate
import functools


def extract_coords(h, *, composition_component="Ti"):
    """
    Extract the phase space coordinate for a given run

    Parameters
    ----------
    h : BlueskyRun
        The run to extract the data from

    composition_component : {'Cu', 'Ti'}, default 'Ti'
        The element to use for the composition component
    """
    return (
        h.metadata["start"][composition_component],
        h.metadata["start"]["anneal_time"],
        h.metadata["start"]["temp"],
    )


def reduce_data(h, peak_locations, *, window_half_width=3):
    """
    Reduce a I(Q) curve to a handful of scalars.

    This sums the intensity in fixed width window around the peak
    locations.

    Parameters
    ----------
    h : BlueskyRun
        The Run to pull the data from

    peak_locations : array[float]
        The location in

    window_half_width : int, default=3
        The half-width of the window.  The window will be
        ``2 * window_half_width + 1`` bins wide approximately centered
        on the peak location.

    Returns
    array
        The scalars extracted from the I(Q) curve.
    """
    p = h.primary.read()
    I = p["I"]
    Q = p["Q"]
    out = []
    indxes = np.searchsorted(Q, peak_locations)
    for indx in indxes:
        peak = np.array(I[indx - window_half_width : indx + window_half_width + 1])
        out.append(np.sum(peak))
    return np.array(out)


def make_full_IofQ_detector(ctrl: Device, *, cat, name: str) -> Device:
    """
    Simulated detector that provides the full I(Q) curve.

    The device that will interpolate the data from *cat* based
    on the position of the SynAxis on *ctrl*.

    This uses `scipy.interpolate.LinearNDInterpolator` to do the
    interpolation which in turn triangulates the input and then uses
    linear barycentric interpolation.

    Parameters
    ----------
    ctrl : Device
        Much have the components *Ti*, *anneal_time*, *temp* and each
        of those must have the component *readback* who's value is a float.
    cat : Catalog
        The source of the experimental data we need to interpolate
    name : str
        The base name of the created device

    Returns
    -------
    Device
        A device with components *I* and *Q*.  *I* is the full
        I(Q) curve interpolated.

    """
    # get the BlueskyRun instances
    data = [cat[uid] for uid in cat]
    # extract the measured control parameters, and setup the
    # interpolation
    full_interpolation = scipy.interpolate.LinearNDInterpolator(
        np.vstack([extract_coords(h) for h in data]),
        np.vstack([h.primary.read()["I"] for h in data]),
    )

    # we are assuming that the Q is the same for all of these!
    Q_data = np.array(data[0].primary.read()["Q"])

    # helper to do the resampling based on the current positions
    # of the controls
    def _resample():
        target = np.array(
            [
                ctrl.Ti.readback.get(),
                ctrl.anneal_time.readback.get(),
                ctrl.temp.readback.get(),
            ]
        )
        return full_interpolation(target).squeeze()

    # define the device class
    class FullI(Device):
        # this closes over the function so it is bound to the ctrl object
        # passed in
        I = Cpt(SynSignal, func=_resample, kind="hinted")
        # this is always the same
        Q = Cpt(SynSignalRO, func=lambda: Q_data, kind="normal")

        # we need to forward the trigger method so that the curve updates
        def trigger(self):
            self.I.trigger()
            return super().trigger()

    # instantiate and return the device
    return FullI(name=name)


def make_ROI_detector(ctrl, peak_locations, reduce_function, *, cat, name):
    """
    Simulated detector that provides ROI values.

    The device that will interpolate the data from *cat* based
    on the position of the SynAxis on *ctrl*.

    This uses `scipy.interpolate.LinearNDInterpolator` to do the
    interpolation which in turn triangulates the input and then uses
    linear barycentric interpolation.

    Parameters
    ----------
    ctrl : Device
        Much have the components *Ti*, *anneal_time*, *temp* and each
        of those must have the component *readback* who's value is a float.

    peak_locations : array[float]
        The locations in Q space to look for features

    reduce_function : Callable[[BlueskyRun, array[float]], array[float]]
        This function should given a BlueskyRun and a list of peak positions
        return as array of floats corresponding to a scalar feature at that
        location.  The signature should be::

            def reduce(h: BlueskyRun, peak_locations : array[float]) -> array[float]:
                ...
    cat : Catalog
        The source of the experimental data we need to interpolate
    name : str
        The base name of the created device

    Returns
    -------
    Device
       The components on this device will depend on the number of peak positions
       given.  For each position, there will be components *I_{NN}* and *Q_{NN}*
       corresponding to the NNth peak location passed in.
    """
    # get the BlueskyRun instances
    data = [cat[uid] for uid in cat]
    # extract the measured control parameters, reduce the data, and setup
    # the interpolation
    reduced_interpolation = scipy.interpolate.LinearNDInterpolator(
        np.vstack([extract_coords(h) for h in data]),
        np.vstack([reduce_data(h, peak_locations) for h in data]),
    )

    # ######
    # this code is too cute for it's own good
    # ######

    # helpers to do the resampling on demand:
    @functools.lru_cache
    def _base_resample(target):
        # target = np.array(target)
        return reduced_interpolation(target).squeeze()

    def _per_peak_resample(indx):
        target = (
            ctrl.Ti.readback.get(),
            ctrl.anneal_time.readback.get(),
            ctrl.temp.readback.get(),
        )
        return _base_resample(target)[indx]

    # define the (variable) number of ROI and peak location components
    # these will re-sample on trigger
    peaks = {
        f"I_{indx:02d}": Cpt(
            SynSignal, func=functools.partial(_per_peak_resample, indx), kind="hinted",
        )
        for indx in range(len(peak_locations))
    }
    # these are fixed
    qs = {
        f"Q_{indx:02d}": Cpt(SynSignalRO, func=lambda x=q: x)
        for indx, q in enumerate(peak_locations)
    }

    # a base class that will forward the trigger call to the I_NN components
    # to do the resampling
    class ForwardTrigger(Device):
        def trigger(self):
            for cpt_name in self.component_names:
                if cpt_name.startswith("I_"):
                    getattr(self, cpt_name).trigger()
            return super().trigger()

    # define the Device class via type
    ROIDetector = type("ROIDetector", (ForwardTrigger,), {**peaks, **qs})

    # instantiate and return the device
    return ROIDetector(name=name)


def make_sim_devices(cat, peak_locations=None):
    if peak_locations is None:
        peak_locations = [
            1.540,  #
            2.665,  # *
            2.945,  # *
            3.077,  # *
            3.549,  #
            3.775,  #
            3.870,  #
            4.614,  # *
            5.019,  #
            5.323,  #
            5.330,  #
        ]

    class Control(Device):
        Ti = Cpt(SynAxis, value=50)
        anneal_time = Cpt(SynAxis, value=30)
        temp = Cpt(SynAxis, value=400)

    ctrl = Control(name="ctrl")

    full = make_full_IofQ_detector(ctrl, cat=cat, name="full")
    rois = make_ROI_detector(
        ctrl, peak_locations, cat=cat, name="rois", reduce_function=reduce_data
    )

    return {obj.name: obj for obj in [ctrl, full, rois]}
