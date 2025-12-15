# BESTEST Validation Report

**Generated:** 2025-12-15 13:59:19

**Standard:** ASHRAE 140-2020 Section 5.2

## Summary

- **Total cases run:** 4
- **Successful:** 4
- **Failed:** 0

## Simulator Capabilities

**Currently Supported Cases:** 600, 620, 900, 920

**Cases Not Yet Implemented** (features under development):

| Case | Reason |
|------|--------|
| 610 | Requires shading geometry (overhang/fins) - approximated with reduced SHGC |
| 630 | Requires shading geometry (overhang/fins) - approximated with reduced SHGC |
| 640 | Requires setback schedules (night setback) |
| 650 | Requires scheduled ventilation (night ventilation) |
| 910 | Requires shading geometry (overhang/fins) - approximated with reduced SHGC |
| 930 | Requires shading geometry (overhang/fins) - approximated with reduced SHGC |
| 940 | Requires setback schedules (night setback) |
| 950 | Requires scheduled ventilation (night ventilation) |

## Results Table

|   Case ID |   Heating (kWh) |   Cooling (kWh) |   Peak Heating (W) |   Peak Cooling (W) |   Max Temp (�C) |   Min Temp (�C) |   Mean Temp (�C) |
|----------:|----------------:|----------------:|-------------------:|-------------------:|----------------:|----------------:|-----------------:|
|       600 |         3601.08 |         202.056 |            45138.2 |            22118.9 |         27.2212 |         19.5486 |          20.8439 |
|       620 |         3601.27 |         202.17  |            45146.1 |            22117.4 |         27.2212 |         19.5485 |          20.8442 |
|       900 |         3516.24 |         117.64  |            45132.5 |            19641.4 |         27.1964 |         19.5487 |          20.8557 |
|       920 |         3516.44 |         117.756 |            45140.9 |            19645.4 |         27.1965 |         19.5486 |          20.8558 |

## Notes

- Heating/Cooling energy in kWh over simulation period
- Peak heating/cooling in Watts
- Temperature statistics in �C
- All case definitions are available in the test suite for future implementation
