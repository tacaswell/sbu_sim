import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
import bluesky.plans as bp
from collections import deque
import itertools


def sort_out_motor(motors, next_point):
    return list(itertools.chain(*zip(motors, next_point)))


class DummyCam:
    def __init__(self, sequence):
        self.sequence = itertools.cycle(sequence)

    def ask(self, n=1):
        return [next(self.sequence) for _ in range(n)]

    def tell(self, *args):
        ...


def adaptive_plan_inter_run(
    dets, motors, adaptive, *, md=None, next_point_callback=None
):
    md = md or {}

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

    if next_point_callback is None:
        next_point_callback = dflt_get_next_point_callback

    (first_point,) = adaptive.ask(1)
    queue.append(first_point)

    @bpp.subs_decorator(next_point_callback)
    def gp_inner_plan():
        uids = []
        while len(queue):
            next_point = queue.pop()
            motor_position_pairs = sort_out_motor(motors, next_point)
            yield from bps.mov(*motor_position_pairs)
            uid = yield from bp.count(dets + motors, md=md)
            uids.append(uid)

        return uids

    return (yield from gp_inner_plan())


def adaptive_plan_intra_run(
    dets, motors, adaptive, *, md=None, next_point_callback=None
):
    md = md or {}

    queue = deque()
    count = 0

    def dflt_get_next_point_callback(name, doc):
        nonlocal count
        # some logic to covert to adaptive food
        # this gets every document, you might want to use
        # DocumentRouter or RunRouter from event_model
        collected_data = ...
        if name == "event" and count < 10:
            adaptive.tell(collected_data)
            (next_point,) = adaptive.ask(1)
            queue.append(next_point)
            count += 1

    if next_point_callback is None:
        next_point_callback = dflt_get_next_point_callback

    (first_point,) = adaptive.ask(1)
    queue.append(first_point)

    @bpp.subs_decorator(next_point_callback)
    @bpp.run_decorator(md=md)
    def gp_inner_plan():
        uids = []
        while len(queue):
            next_point = queue.pop()
            motor_position_pairs = sort_out_motor(motors, next_point)
            yield from bps.mov(*motor_position_pairs)
            uid = yield from bps.trigger_and_read(dets + motors)
            uids.append(uid)

        return uids

    return (yield from gp_inner_plan())
