# BESTEST Validation Report

**Generated:** 2026-01-13 17:08:36

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
|       600 |         4075.81 |         7018.5  |            3144.35 |            6832.98 |         27.6833 |         19.6856 |          23.3956 |
|       620 |         4193.96 |         5722.61 |            3233.38 |            4898.9  |         27.4899 |         19.6767 |          23.4219 |
|       900 |         1236.58 |         3793.07 |            2659.26 |            3561.13 |         27.3561 |         19.7341 |          24.5883 |
|       920 |         2528.12 |         4338.01 |            2641.06 |            3568.54 |         27.3569 |         19.7359 |          23.7733 |

## Notes

- Heating/Cooling energy in kWh over simulation period
- Peak heating/cooling in Watts
- Temperature statistics in �C
- All case definitions are available in the test suite for future implementation
