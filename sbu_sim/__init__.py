from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
from . import ticu


def initialize(user_ns, broker_name, reduced_cat_name):
    """
    Initialize an interactive name space for use.

    Parameters
    ----------
    user_ns : dict
        The nampespace to populate.  Typically ``get_ipython().user_ns``

    broker_name : str
        The name of the databroker to insert into

    reduced_cat_name : str
        The catalog that has the reduced TiCu data available
    """
    import nslsii
    import databroker

    ret = nslsii.configure_base(
        user_ns, broker_name, configure_logging=False, ipython_exc_logging=False
    )
    user_ns["db"] = user_ns["db"].v2
    ticu_sim = ticu.make_sim_devices(databroker.catalog[reduced_cat_name])
    if set(user_ns).intersection(ticu_sim):
        overlap = set(user_ns).intersection(ret)
        raise ValueError(f"there are overlapping names {overlap}")

    user_ns.update(ticu_sim)

    return ret + list(ticu_sim)
