import pandas as pd

# Academic period labels for FS ČVUT, extracted from official schedules.
# Periods are right-exclusive on end date (end = first day NOT in period).
# Data outside these ranges gets label "unknown".
# Years 2021/2022 – 2023/2024 not yet provided → "unknown" for that range.

_PERIODS = [
    # 2016/2017
    ("teaching_winter",      "2016-10-03", "2016-12-23"),
    ("christmas_break",      "2016-12-23", "2017-01-02"),
    ("exam_winter",          "2017-01-12", "2017-02-18"),
    ("teaching_summer",      "2017-02-20", "2017-05-29"),
    ("exam_summer",          "2017-05-29", "2017-07-01"),
    ("summer_break",         "2017-07-03", "2017-09-04"),
    ("extraordinary_exam",   "2017-09-04", "2017-09-16"),

    # 2017/2018
    ("teaching_winter",      "2017-10-01", "2017-12-23"),
    ("christmas_break",      "2017-12-23", "2018-01-02"),
    ("exam_winter",          "2018-01-11", "2018-02-17"),
    ("teaching_summer",      "2018-02-19", "2018-05-28"),
    ("exam_summer",          "2018-05-28", "2018-06-30"),
    ("summer_break",         "2018-07-02", "2018-09-03"),
    ("extraordinary_exam",   "2018-09-03", "2018-09-15"),

    # 2018/2019
    ("teaching_winter",      "2018-10-01", "2018-12-23"),
    ("christmas_break",      "2018-12-23", "2019-01-02"),
    ("exam_winter",          "2019-01-09", "2019-02-18"),  # summer semester starts 18.2, exams officially to 22.2
    ("teaching_summer",      "2019-02-18", "2019-05-28"),
    ("exam_summer",          "2019-05-28", "2019-06-29"),
    ("summer_break",         "2019-07-03", "2019-09-03"),
    ("extraordinary_exam",   "2019-09-03", "2019-09-14"),

    # 2019/2020
    ("teaching_winter",      "2019-09-23", "2019-12-23"),
    ("christmas_break",      "2019-12-23", "2020-01-06"),
    ("exam_winter",          "2020-01-06", "2020-02-08"),
    ("teaching_summer",      "2020-02-10", "2020-05-18"),
    ("exam_summer",          "2020-05-18", "2020-06-27"),
    ("summer_break",         "2020-06-29", "2020-08-31"),
    ("extraordinary_exam",   "2020-08-31", "2020-09-12"),

    # 2020/2021
    ("teaching_winter",      "2020-09-21", "2020-12-21"),
    ("christmas_break",      "2020-12-21", "2021-01-04"),
    ("exam_winter",          "2021-01-07", "2021-02-13"),
    ("teaching_summer",      "2021-02-15", "2021-05-21"),
    ("exam_summer",          "2021-05-21", "2021-06-28"),
    ("summer_break",         "2021-06-28", "2021-08-30"),
    ("extraordinary_exam",   "2021-08-30", "2021-09-11"),

    # 2021/2022
    ("teaching_winter",      "2021-09-20", "2021-12-20"),
    ("christmas_break",      "2021-12-20", "2022-01-03"),
    ("exam_winter",          "2022-01-06", "2022-02-12"),
    ("teaching_summer",      "2022-02-14", "2022-05-20"),
    ("exam_summer",          "2022-05-20", "2022-06-25"),
    ("summer_break",         "2022-06-27", "2022-09-01"),
    ("extraordinary_exam",   "2022-09-01", "2022-09-10"),

    # 2022/2023
    ("teaching_winter",      "2022-09-19", "2022-12-19"),
    ("christmas_break",      "2022-12-19", "2023-01-09"),
    ("exam_winter",          "2023-01-16", "2023-02-18"),
    ("teaching_summer",      "2023-02-20", "2023-05-29"),
    ("exam_summer",          "2023-05-29", "2023-07-01"),
    ("summer_break",         "2023-07-03", "2023-09-04"),
    ("extraordinary_exam",   "2023-09-04", "2023-09-16"),

    # 2023/2024
    ("teaching_winter",      "2023-09-25", "2023-12-23"),  # teaching ends Dec 22, Christmas starts Dec 25
    ("christmas_break",      "2023-12-25", "2024-01-08"),
    ("exam_winter",          "2024-01-08", "2024-02-19"),
    ("teaching_summer",      "2024-02-19", "2024-05-27"),
    ("exam_summer",          "2024-05-27", "2024-06-29"),
    ("summer_break",         "2024-07-01", "2024-09-02"),
    ("extraordinary_exam",   "2024-09-02", "2024-09-14"),

    # 2024/2025  (from Casovy_plan_24_25.pdf)
    ("teaching_winter",      "2024-09-23", "2024-12-21"),
    ("christmas_break",      "2024-12-23", "2025-01-06"),
    ("exam_winter",          "2025-01-06", "2025-02-15"),
    ("teaching_summer",      "2025-02-17", "2025-05-24"),
    ("exam_summer",          "2025-05-26", "2025-06-28"),
    ("summer_break",         "2025-06-30", "2025-09-01"),
    ("extraordinary_exam",   "2025-09-01", "2025-09-13"),

    # 2025/2026  (from Casovy_plan_25_26.pdf)
    ("teaching_winter",      "2025-09-22", "2025-12-20"),
    ("christmas_break",      "2025-12-22", "2026-01-05"),
    ("exam_winter",          "2026-01-05", "2026-02-14"),
    ("teaching_summer",      "2026-02-16", "2026-05-23"),
    ("exam_summer",          "2026-05-25", "2026-06-27"),
    ("summer_break",         "2026-06-29", "2026-08-31"),
    ("extraordinary_exam",   "2026-08-31", "2026-09-12"),
]


def label_academic_period(index: pd.DatetimeIndex) -> pd.Series:
    """Map a DatetimeIndex to academic period labels.

    Returns a Series aligned to `index` with string labels.
    Timestamps outside the covered range are labelled 'unknown'.
    """
    labels = pd.Series("unknown", index=index, dtype="object", name="academic_period")
    dates = index.normalize()  # drop time component for range comparison

    tz = index.tz
    for label, start, end in _PERIODS:
        mask = (dates >= pd.Timestamp(start, tz=tz)) & (dates < pd.Timestamp(end, tz=tz))
        labels[mask] = label

    return labels
