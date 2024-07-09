from datetime import timedelta, datetime


def normalize_datetime(date, seconds_per_dt, ref_time):
    """Normalizes datetime object for ML input.

    Time starts at 2010-01-01 with max time range == 2 pi
    Parameters
    ----------
    date: input date

    Returns
    -------
    normalized date
    """
    return (date - ref_time).total_seconds() / seconds_per_dt


def unnormalize_datetime(norm_date: float, seconds_per_dt, ref_time) -> datetime:
    """Computes the actual datetime from a normalized date.

    Parameters
    ----------
    norm_date: normalized date

    Returns
    -------
    real datetime
    """
    return ref_time + timedelta(seconds=norm_date * seconds_per_dt)
