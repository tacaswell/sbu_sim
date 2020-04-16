=======
sbu_sim
=======

.. image:: https://img.shields.io/travis/nslsii/sbu_sim.svg
        :target: https://travis-ci.org/nslsii/sbu_sim

.. image:: https://img.shields.io/pypi/v/sbu_sim.svg
        :target: https://pypi.python.org/pypi/sbu_sim


Provide simulated detectors for reduced data from XPD for Chen-Wiegart group at SBU

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!)

To use ::

  import sbu_sim
  # change the third argument to where you unpacked the TiCu data
  sbu_sim.initialize(get_ipython().user_ns, 'temp', 'xpd_auto_202003_msgpack');
  # run a scan varying the titanium and getting back area-under-peak values
  RE(bp.scan([rois, ctrl], ctrl.Ti, 0, 100, 25))
  # run a scan varying the titanium and getting back the full I(Q) at each point
  RE(bp.scan([full, ctrl], ctrl.Ti, 0, 100, 25))


To get the TiCu data you will need to get a copy of the packed data (left
as an exercise for the reader).  Once you have unzipped it do

.. code:: bash

   pip install databroker-pack
   databroker-unpack path/to/unpacked/tarbal xpd_auto_202003_msgpack


See https://blueskyproject.io/databroker-pack/
