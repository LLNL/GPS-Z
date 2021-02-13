

import os

SERIAL = bool(os.getenv("GPS_SERIAL",default=False))

if SERIAL:
    from src.gui.def_run_serial import *
else:
    from src.gui.def_run_pool import *

