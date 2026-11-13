"""JPL Horizons position lookup that does not depend on Horizons' CSV output.

Drop-in replacement for ``sunpy.coordinates.get_horizons_coord``. Since November 2026
the Horizons CSV vector table (``CSV_FORMAT=YES``) omits the X/Y/Z columns, which makes
the SunPy implementation fail with ``KeyError: 'X'``. This version requests the same
state vectors in the labeled text format and parses them directly.
"""
import re

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, HeliocentricEclipticIAU76, SkyCoord
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst
from sunpy.time import parse_time

HORIZONS_FILE_API = "https://ssd.jpl.nasa.gov/api/horizons_file.api"

_ID_PREFIX = {
    None: "",
    "smallbody": "",
    "designation": "DES=",
    "name": "NAME=",
    "asteroid_name": "ASTNAM=",
    "comet_name": "COMNAM=",
}
_NUMBER = r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)"
_EPOCH_RE = re.compile(r"^\s*(\d+\.\d+)\s*=\s*A\.D\.")
_POSITION_RE = re.compile(rf"\bX\s*=\s*{_NUMBER}\s*Y\s*=\s*{_NUMBER}\s*Z\s*=\s*{_NUMBER}")
_VELOCITY_RE = re.compile(rf"\bVX\s*=\s*{_NUMBER}\s*VY\s*=\s*{_NUMBER}\s*VZ\s*=\s*{_NUMBER}")


def _query(body, time, id_type):
    if id_type not in _ID_PREFIX:
        raise ValueError("Invalid id_type")
    if id_type == "smallbody" and not body.endswith(";"):
        body += ";"
    args = {
        "COMMAND": f"'{_ID_PREFIX[id_type]}{body}'",
        "EPHEM_TYPE": "VECTORS",
        "OUT_UNITS": "AU-D",
        "CENTER": "500@10",
        "VEC_TABLE": "2",
        "VEC_LABELS": "YES",
        "CSV_FORMAT": "NO",
    }
    if isinstance(time, dict):
        if set(time.keys()) != {"start", "stop", "step"}:
            raise ValueError('time dictionary must have the keys ["start", "stop", "step"]')
        jpl_fmt = "'%Y-%m-%d %H:%M:%S.%f'"
        args["START_TIME"] = parse_time(time["start"]).tdb.strftime(jpl_fmt)
        args["STOP_TIME"] = parse_time(time["stop"]).tdb.strftime(jpl_fmt)
        args["STEP_SIZE"] = time["step"]
    else:
        epochs = np.reshape(time, (-1,)).tdb.jd
        if epochs.size >= 10000:
            raise ValueError("For more than 10,000 time values, use dictionary input.")
        if epochs.size == 1:
            # Horizons currently fails on single-epoch TLIST requests; pad with a later epoch
            # that sorts last in the output and is dropped by the caller.
            epochs = np.append(epochs, epochs[0] + 1.0)
        args["TLIST"] = "\n".join(f"{epoch:.9f}" for epoch in epochs)
        args["TLIST_TYPE"] = "JD"

    contents = "!$$SOF\n" + "\n".join(f"{k}={v}" for k, v in args.items())
    response = requests.post(HORIZONS_FILE_API, data={"format": "text"}, files={"input": contents}, timeout=300)
    response.raise_for_status()
    return response.text


def _parse(text, body, include_velocity):
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("$$SOE")) + 1
        stop = next(i for i, line in enumerate(lines) if line.startswith("$$EOE"))
    except StopIteration:
        raise RuntimeError(f"JPL Horizons returned no ephemeris for {body!r}:\n{text}") from None

    block = "\n".join(lines[start:stop])
    epochs = [float(m.group(1)) for line in lines[start:stop] if (m := _EPOCH_RE.match(line))]
    position = np.array(_POSITION_RE.findall(block.replace("D", "E")), dtype=np.float64).reshape(-1, 3)
    velocity = np.array(_VELOCITY_RE.findall(block.replace("D", "E")), dtype=np.float64).reshape(-1, 3)
    if len(epochs) == 0 or position.shape[0] != len(epochs) or (include_velocity and velocity.shape[0] != len(epochs)):
        raise RuntimeError(
            f"Could not parse JPL Horizons vectors for {body!r}: "
            f"{len(epochs)} epochs, {position.shape[0]} positions, {velocity.shape[0]} velocities"
        )
    return np.asarray(epochs), position, velocity


def get_horizons_coord(body, time="now", id_type=None, *, include_velocity=False):
    """Return a HeliographicStonyhurst SkyCoord for ``body`` from JPL Horizons.

    Same call signature and output as ``sunpy.coordinates.get_horizons_coord``.
    """
    if isinstance(time, dict):
        text = _query(body, time, id_type)
        epochs, position, velocity = _parse(text, body, include_velocity)
        obstime = Time(parse_time(epochs, format="jd", scale="tdb"), format="isot", scale="utc")
    else:
        obstime = parse_time(time)
        # Horizons rejects a TLIST with repeated epochs and sorts its output by epoch: query the sorted unique
        # epochs once and map them back to the requested order.
        flat = np.reshape(obstime, (-1,))
        _, first, inverse = np.unique(flat.tdb.jd, return_index=True, return_inverse=True)
        text = _query(body, flat[first], id_type)
        epochs, position, velocity = _parse(text, body, include_velocity)
        if first.size == 1 and epochs.size == 2:
            epochs, position, velocity = epochs[:1], position[:1], velocity[:1]
        if epochs.size != first.size:
            raise RuntimeError(f"JPL Horizons returned {epochs.size} epochs for {first.size} requested times")
        position = position[inverse]
        velocity = velocity[inverse] if include_velocity else velocity

    vector = CartesianRepresentation(position.T * u.AU)
    if include_velocity:
        vector = vector.with_differentials(CartesianDifferential(velocity.T * u.AU / u.day))
    coord = SkyCoord(vector, frame=HeliocentricEclipticIAU76, obstime=np.reshape(obstime, (-1,)))
    return coord.transform_to(HeliographicStonyhurst).reshape(obstime.shape)
