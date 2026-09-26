# Drift Model Validation Against the Real MV Wakashio Incident

Wreck position: -20.4442, 57.7433 (Pointe d'Esny reef, SE Mauritius)
Real oil-leak start: 2020-08-06T06:00:00+00:00 (hull crack date, not the 25 Jul grounding)

## Published ground truth

- Sasamal & Kalyan (2021), Marine Pollution Bulletin -- GNOME simulation: oil drifted westward, reached Pointe d'Esny shore in 2h30m, continued the same direction through the first 6h. (https://www.sciencedirect.com/science/article/abs/pii/S0025326X21009267)
- Assessment of MV Wakashio oil spill via satellite imagery, J. Earth Syst. Sci. (2022) -- northwestward drift from wind/Stokes drift/tides; deposition along ~28 km of coastline from Pointe d'Esny; no southward movement observed. (https://link.springer.com/article/10.1007/s12040-021-01763-3)
- INCOIS HYCOM-based trajectory model for the Mauritius domain, reported in good agreement with real Sentinel-1A slick extent. (https://www.ias.ac.in/public/Volumes/jess/131/00/0042.pdf)

Published drift direction range used for comparison: 270.0°-340.0° (west through north-northwest).
Published time-to-shore: 2.5-6.0 hours.

## This project's own forecast, run from the real position/time

- Predicted bearing: **35.6°**
- Predicted distance over the forecast window: 1.31 km
- Data source actually used: `synthetic_wind+synthetic_current`
- Direction check: **DOES NOT MATCH** the published range

## Honest reading

Ran on the OFFLINE SYNTHETIC CLIMATOLOGY FALLBACK, not live reanalysis data (no internet route to Open-Meteo in this environment). The fallback's fixed current constants dominate over the small (3%) wind-leeway term in this project's drift physics, and for this specific incident that fallback current happens to point the wrong way (net drift is northeast-ish, not the published northwest). This is a genuine limitation of the offline fallback for this incident, not evidence about the live-data path's accuracy -- re-run this validation with real internet access (so the historical wind/current APIs are actually reached) before drawing conclusions about live-data accuracy.

This is disclosed rather than hidden because a validation that only reports success
when it succeeds isn't a validation. The mechanism -- running the real forecast
function against real cited ground truth and checking numerically, not by eye -- is
sound and reusable; what it currently reveals is a real gap in the offline fallback's
applicability to this specific incident, and a concrete next step (re-run with live
historical weather data) rather than an untested claim either way.