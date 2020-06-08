"""Tools to integrate adaptive decsion making with Bluesky."""
from collections import deque
import itertools
from queue import Queue

import numpy as np

import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
import bluesky.plans as bp


def chain_zip(motors, next_point):
    """
    Interlace motors and next_point values.

    This converts two lists ::

       m = (motor1, motor2, ..)
       p = (pos1, pos2, ..)

    into a single list ::

       args = (motor1, pos1, motor2, pos2, ...)

    which is what `bluesky.plan_stubs.mv` expects.

    Parameters
    ----------
    motors : iterable
        The list of ophyd objects that should be moved,
        must be same length and next_point

    next_point : iterable
        The list values, must be same length as motors.

    Returns
    -------
    list
       A list alternating (motor, value, motor, value, ...)

    """
    return list(itertools.chain(*zip(motors, next_point)))


def extract_arrays(endogenous_keys, exogenous_keys, payload):
    """
    Extract the endogenous and exogenous data from Event['data'].

    Parameters
    ----------
    endogenous_keys : List[str]
        The names of the endogenous keys in the events

    exogenous_keys : List[str]
        The names of the exogenous keys in the events
    payload : dict[str, Any]
        The ev['data'] dict from an Event Model Event document.

    Returns
    -------
    independent : np.array
        A numpy array where the first axis maps to the endogenous variables

    measurements : np.array
        A numpy array where the first axis maps to the exogenous variables

    """
    independent = np.asarray([payload[k] for k in endogenous_keys])
    # This is the extracted measurements
    measurement = np.asarray([payload[k] for k in exogenous_keys])
    return independent, measurement


# These tools are for integrating the adaptive logic inside of a run.
# They are expected to get single events and provide feedback to drive
# the plan based in that information.  This is useful when the computation
# time to make the decision is short compared to acquisition / movement time
# and the computation is amenable to streaming analysis.
#
# This is a "fine" grained integration of the adaptive logic into data acquisition.


def intra_plan_sequence_factory(
    sequence, endogenous_keys, exogenous_keys, *, max_count=10, queue=None
):
    """
    Generate the callback and queue for a naive recommendation engine.

    This returns the same sequence of points no matter what the
    measurements are.

    For each Event that the callback sees it will place either a
    recommendation or `None` into the queue.  Recommendations will be
    of a dict mapping the endogenous_keys to the recommended values and
    should be interpreted by the plan as a request for more data.  A `None`
    placed in the queue should be interpreted by the plan as in instruction to
    terminate the run.


    Parameters
    ----------
    sequence : iterable of positions
        This should be an iterable of positions vectors that match the motors

    endogenous_keys : List[str]
        The names of the endogenous keys in the events

    exogenous_keys : List[str]
        The names of the exogenous keys in the events

    max_count : int, optional
        The maximum number of measurements to take before poisoning the queue.

    queue : Queue, optional
        The communication channel for the callback to feedback to the plan.
        If not given, a new queue will be created.

    Returns
    -------
    callback : Callable[str, dict]
        This function must be subscribed to RunEngine to receive the
        document stream.

    queue : Queue
        The communication channel between the callback and the plan.  This
        is always returned (even if the user passed it in).

    """
    seq = iter(itertools.cycle(sequence))
    if queue is None:
        queue = Queue()

    # TODO handle multi-stream runs!
    def callback(name, doc):

        if name == "event":
            if doc["seq_num"] >= max_count:
                # if at max number of points poison the queue and return early
                queue.put(None)
                return
            payload = doc["data"]
            inp = np.asarray([payload[k] for k in endogenous_keys])
            measurement = np.asarray([payload[k] for k in exogenous_keys])
            # call something to get next point!
            next_point = next(seq)
            queue.put({k: v for k, v in zip(endogenous_keys, next_point)})

    return callback, queue


def intra_plan_step_factory(
    step, endogenous_keys, exogenous_keys, *, max_count=10, queue=None
):
    """
    Generate the callback and queue for a naive recommendation engine.

    This recommends a fixed step size independent of the measurement.

    For each Event that the callback sees it will place either a
    recommendation or `None` into the queue.  Recommendations will be
    of a dict mapping the endogenous_keys to the recommended values and
    should be interpreted by the plan as a request for more data.  A `None`
    placed in the queue should be interpreted by the plan as in instruction to
    terminate the run.


    Parameters
    ----------
    step : array[float]
        The delta step to take on each point

    endogenous_keys : List[str]
        The names of the endogenous keys in the events

    exogenous_keys : List[str]
        The names of the exogenous keys in the events

    max_count : int, optional
        The maximum number of measurements to take before poisoning the queue.

    queue : Queue, optional
        The communication channel for the callback to feedback to the plan.
        If not given, a new queue will be created.

    Returns
    -------
    callback : Callable[str, dict]
        This function must be subscribed to RunEngine to receive the
        document stream.

    queue : Queue
        The communication channel between the callback and the plan.  This
        is always returned (even if the user passed it in).

    """
    if queue is None:
        queue = Queue()

    def callback(name, doc):

        # TODO handle multi-stream runs!
        if name == "event":
            if doc["seq_num"] > max_count:
                # if at max number of points poison the queue and return early
                queue.put(None)
                return
            payload = doc["data"]
            # This is your "motor positions"
            independent = np.asarray([payload[k] for k in endogenous_keys])
            # This is the extracted measurements
            measurement = np.asarray([payload[k] for k in exogenous_keys])
            # call something to get next point!
            next_point = independent + step
            queue.put({k: v for k, v in zip(endogenous_keys, next_point)})

    return callback, queue


def intra_plan_gpcam_factory(
    gpcam_object, endogenous_keys, exogenous_keys, *, max_count=10, queue=None
):
    """
    Generate the callback and queue for gpCAM integration.

    For each Event that the callback sees it will place either a
    recommendation or `None` into the queue.  Recommendations will be
    of a dict mapping the endogenous_keys to the recommended values and
    should be interpreted by the plan as a request for more data.  A `None`
    placed in the queue should be interpreted by the plan as in instruction to
    terminate the run.


    Parameters
    ----------
    pgcam_object : gpCAM
        The gpcam recommendation engine

    endogenous_keys : List[str]
        The names of the endogenous keys in the events

    exogenous_keys : List[str]
        The names of the exogenous keys in the events

    max_count : int, optional
        The maximum number of measurements to take before poisoning the queue.

    queue : Queue, optional
        The communication channel for the callback to feedback to the plan.
        If not given, a new queue will be created.

    Returns
    -------
    callback : Callable[str, dict]
        This function must be subscribed to RunEngine to receive the
        document stream.

    queue : Queue
        The communication channel between the callback and the plan.  This
        is always returned (even if the user passed it in).

    """
    if queue is None:
        queue = Queue()

    # TODO handle multi-stream runs!
    def callback(name, doc):
        if name != "event":
            return

        if doc["seq_num"] > max_count:
            # if at max number of points, return early
            queue.put(None)
            return
        independent, measurement = extract_arrays(
            endogenous_keys, exogenous_keys, doc["data"]
        )
        # # GPCAM CODE GOES HERE
        raise NotImplementedError
        gpcam_object.tell(independent, measurement)
        next_point = gpcam_object.ask(1)
        # # GPCAM CODE GOES HERE
        queue.put({k: v for k, v in zip(endogenous_keys, next_point)})

    return callback, queue


def intra_run_adaptive_plan(
    dets,
    first_point,
    *,
    to_brains,
    from_brains,
    md=None,
    take_reading=bps.trigger_and_read
):
    """
    Execute an adaptive scan using an intra-run recommendation engine.

    Parameters
    ----------
    dets : List[OphydObj]
       The detector to read at each point.  The exogenous keys that the
       recommendation engine is looking for must be provided by these
       devices.

    first_point : Dict[Settable, Any]
       The first point of the scan.  The motors that will be scanned
       are extracted from the keys.  The endogenous keys that the
       recommendation engine is looking for / returning must be provided
       by these devices.

    to_brains : Callable[str, dict]
       This is the callback that will be registered to the RunEngine.

       The expected contract is for each event it will place either a
       dict mapping endogenous variable to recommended value or None.

       This plan will either move to the new position and take data
       if the value is a dict or end the run if `None`

    from_brains : Queue
       The consumer side of the Queue that the recommendation engine is
       putting the recommendations onto.

    md : dict[str, Any], optional
       Any extra meta-data to put in the Start document

    take_reading : plan, optional
        function to do the actual acquisition ::

           def take_reading(dets, name='primary'):
                yield from ...

        Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

        Defaults to `trigger_and_read`
    """
    # TODO inject args / kwargs here.
    _md = {"hints": {}}
    _md.update(md or {})
    try:
        _md["hints"].setdefault(
            "dimensions", [(m.hints["fields"], "primary") for m in first_point.keys()]
        )
    except (AttributeError, KeyError):
        ...

    # extract the motors
    motors = list(first_point.keys())
    # convert the first_point variable to from we will be getting
    # from queue
    first_point = {m.name: v for m, v in first_point.items()}

    @bpp.subs_decorator(to_brains)
    @bpp.run_decorator(md=_md)
    def gp_inner_plan():
        next_point = first_point
        while True:
            # this assumes that m.name == the key in Event
            target = {m: next_point[m.name] for m in motors}
            motor_position_pairs = itertools.chain(*target.items())
            yield from bps.mov(*motor_position_pairs)
            yield from take_reading(dets + motors)
            next_point = from_brains.get(timeout=1)
            if next_point is None:
                return

    return (yield from gp_inner_plan())


# These functions are for integrating adaptive logic that works
# between runs.  The decision making process can expect to consume a
# full run before having to make a suggestion about what to do next.
# This may be desirable if there is a major time miss-match between
# the computation and the experiment, of the data collection is not
# amenable to streaming analysis, or the natural structure of the
# experiment dictates.
#
# This corresponds to a "middle" scale of adaptive integration into
# data collection.


def inter_run_factory_sequence(endogenous_keys, exogenous_keys, *, max_count=10):
    queue = deque()
    count = 0

    def dflt_get_next_point_callback(name, doc):
        nonlocal count
        # some logic to covert to adaptive food
        # this gets every document, you might want to use
        # DocumentRouter or RunRouter from event_model
        collected_data = ...
        if name == "stop" and count < 10:
            adaptive.tell(collected_data)
            (next_point,) = adaptive.ask(1)
            queue.append(next_point)
            count += 1

    return dflt_get_next_point_callback, queue


def adaptive_inter_run_scan(dets, motors, adaptive, *, md=None, callback, queue):
    md = md or {}
    count = 0

    if next_point_callback is None:
        next_point_callback = dflt_get_next_point_callback

    (first_point,) = adaptive.ask(1)
    queue.append(first_point)

    @bpp.subs_decorator(next_point_callback)
    def gp_inner_plan():
        uids = []
        while len(queue):
            next_point = queue.pop()
            motor_position_pairs = chain_zip(motors, next_point)
            yield from bps.mov(*motor_position_pairs)
            uid = yield from bp.count(dets + motors, md=md)
            uids.append(uid)

        return uids

    return (yield from gp_inner_plan())
