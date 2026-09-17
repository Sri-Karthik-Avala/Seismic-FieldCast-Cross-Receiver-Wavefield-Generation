# Seismic FieldCast: Cross-Receiver Wavefield Generation

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Sequence To Sequence |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 2 |
| Last submission | 2026-09-06 |

## Problem statement

### Overview

A seismic network records the same earthquake at multiple ground stations. In

this challenge, a **receiver** means one seismic station. Four stations provide

simultaneous measurements of an earthquake, while the measurement from a fifth

station is withheld. Your task is to generate the signal measured at that

target station from the four support-station signals and the relative geometry

of all five stations.

The measurements come from the Southern California Earthquake Data Center

(SCEDC) and the Caltech/USGS Southern California Seismic Network (SCSN),

network code `CI`. The original recordings are continuous vertical-component

`BHZ` seismometer streams sampled at 40 Hz. Earthquake origin times from the

SCSN catalog define synchronized 60-second windows running from 10 seconds

before to 50 seconds after each catalog origin.

The released values are not raw waveform counts. Every station recording is

converted into three normalized amplitude-envelope bands:

- **low band:** 1–3 Hz;
- **middle band:** 3–6 Hz;
- **high band:** 6–12 Hz.

For each band, the waveform is bandpass filtered, converted to an

analytic-signal magnitude, smoothed, normalized by the station-and-band

background level, compressed with `log1p`, and sampled at 10 frames per second.

One station therefore produces a nonnegative array with shape `[3,600]`: three

frequency bands by 600 time frames. These values describe relative seismic

amplitude through time; they are not calibrated displacement, velocity, or

earthquake magnitude.

Each row provides four station fields in one NumPy array with shape

`[4,3,600]`. The hidden target is the corresponding `[3,600]` field measured at

a fifth station during the same earthquake and the same 60-second interval.

The model must reproduce arrival timing, bandwise amplitude, secondary

arrivals, duration, and decay at the target station.

This is not ordinary time-series forecasting. No future interval follows the

input interval: support and target cover the same time window. The task is

cross-station spatial reconstruction of a real earthquake signal.

The targets are not simulator output, synthetic mixtures, or semantic labels.

Every target is derived from a real held-out station recording using exactly

the same deterministic transformation as the four inputs.

The split tests two kinds of generalization at once. Private events come from event periods absent from training, and private target-receiver roles do not occur as public training targets. Rows sharing an `event_group` come from the same event and must remain together during local validation. Every target is a fifth receiver distinct from the four support receivers supplied for that row.

Public receiver geometry is expressed in kilometres in a different anonymous Cartesian frame for every row. It is not GPS or a global map. Absolute location, orientation, exact elevation datum, source timestamps, stable receiver identities, and the original network geometry are not exposed.

During platform preparation, a key derived from private-only evaluation

material applies a second deterministic privacy transform. Row ids and event

groups are remapped and rows are permuted. The four public support arrays

receive small band-specific scale/offset changes and low-amplitude noise, while

their 600-frame timing is preserved. Receiver coordinates receive another

rigid transform, translation, and small perturbation. Training targets and

private evaluation targets are not numerically altered: scoring remains tied

to the measured target-receiver representation.

### Intended Approach

Use local or open scientific-ML and signal-processing methods on the provided training pairs. This is a CPU-only challenge: submitted solutions must be practical without GPU access and should fit within the 1.5-hour solution limit on 10 CPU cores and 62 GB RAM.

Valid solutions may combine geometry-aware interpolation, learned delay and gain estimation, temporal convolution, frequency-domain transfer features, low-rank event representations, coordinate-conditioned decoders, tree or linear residual models, nearest-receiver methods, and calibrated ensembles.

Use grouped validation by `event_group`. A random row split places alternate target views of the same event in both folds and gives an unrealistically optimistic score.

### What Not To Use

Do not use upstream archive lookup, public event-catalog search, waveform or array fingerprinting, source-period reconstruction, receiver-network matching, global-location recovery, filename or row-order lookup, file metadata, private files, hard-coded answer dictionaries, manual hidden-test labeling, hosted or closed-source seismic APIs, or grader and filesystem exploitation.

Do not treat opaque ids as features. Do not attempt to reverse the per-row anonymous coordinate transformation. Solve from the released support arrays, anonymous geometry, training targets, and public CSV fields only.

### Enforcement On Invalid Approaches

External-retrieval submissions, metadata-only submissions, source or receiver reconstruction, private-file probes, hard-coded mappings, manual test labeling, or approaches that do not solve the cross-receiver generation task may be rejected before payout regardless of leaderboard score.

### Evaluation

Scores are maximized and range from `0.0` to `1.0`.

For row `i`, let `Y_i` be the true target field and `P_i` the submitted field, both with shape `[3,600]`.

```
RMSE_i  = sqrt(mean((P_i - Y_i)^2))

Scale_i = sqrt(mean(Y_i^2))

Skill_i = max(0, 1 - RMSE_i / max(Scale_i, 1e-12))

FinalScore = mean(Skill_i)
```

The normalization makes the metric relative to the true event energy. A perfect submission scores exactly `1.0`. A zero-field submission scores `0.0`. Predictions worse than the zero-field reference are clipped to `0.0`.

The grader raises an invalid-submission error for missing, extra, or reordered columns; duplicate ids; or an id set different from `test.csv`. Row-local malformed JSON, boolean values, non-finite values, values outside `[0,30]`, or arrays with a shape other than `[3,600]` give that row a score of `0.0` rather than crashing the grader.

### Dataset

The public data contains CSV files and NumPy support arrays. Training rows

include measured target fields; test rows contain only inputs.

| Item | Description |

|---|---|

| `public/supports/train/*.npy` | Training support arrays |

| `public/supports/test/*.npy` | Test support arrays |

| `public/train.csv` | Training inputs and target fields |

| `public/test.csv` | Test inputs only |

| `public/sample_submission.csv` | Valid placeholder submission |

| `public/wavefield_schema.json` | Machine-readable array schema |

The prepared split contains 165 training rows from 55 earthquake groups and 64

test rows from 32 earthquake groups. Three training targets or two test targets

are constructed for each earthquake. Rows with the same `event_group` contain

different target stations for the same underlying event.

`train.csv` Columns

| Column | Type | Description |

|---|---|---|

| `id` | string | Salted opaque row identifier |

| `event_group` | string | Opaque identifier shared by rows from the same earthquake |

| `support_path` | string | Relative path to the row's `[4,3,600]` NumPy support array |

| `support_receivers_json` | string | JSON list of four support-coordinate objects aligned with support-array axis 0 |

| `target_x_km` | float | Target x-coordinate in the row-local anonymous frame |

| `target_y_km` | float | Target y-coordinate in the row-local anonymous frame |

| `target_z_km` | float | Target relative vertical coordinate in the row-local anonymous frame |

| `frame_rate_hz` | int | Temporal frame rate; always 10 |

| `n_frames` | int | Number of frames; always 600 |

| `wavefield_json` | string | Measured target field as a numeric JSON array with shape `[3,600]` |

`support_receivers_json` contains exactly four objects. Object `j` describes

`support[j,:,:]` and has this structure:

```
{"receiver":"support_0","x_km":-8.2143,"y_km":3.1187,"z_km":-0.0612}
```

The displayed numbers are illustrative. `receiver` is a row-local array label,

not a persistent station identity.

`test.csv` Columns

| Column | Type | Description |

|---|---|---|

| `id` | string | Salted opaque row identifier |

| `event_group` | string | Opaque identifier shared by rows from the same earthquake |

| `support_path` | string | Relative path to the row's `[4,3,600]` NumPy support array |

| `support_receivers_json` | string | JSON list of four support-coordinate objects aligned with support-array axis 0 |

| `target_x_km` | float | Target x-coordinate in the row-local anonymous frame |

| `target_y_km` | float | Target y-coordinate in the row-local anonymous frame |

| `target_z_km` | float | Target relative vertical coordinate in the row-local anonymous frame |

| `frame_rate_hz` | int | Temporal frame rate; always 10 |

| `n_frames` | int | Number of frames; always 600 |

`test.csv` has the same input columns as `train.csv` and omits only

`wavefield_json`.

### Receiver Geometry

The x, y, and z values are kilometres in a different anonymous Cartesian

coordinate frame for every row. Support and target coordinates within one row

share that frame, so relative distances and directions are meaningful. They

are not latitude, longitude, GPS coordinates, absolute elevation, or stable

station locations, and coordinates from different rows must not be merged.

### Array Alignment

Support array axes are `[support_receiver, frequency_band, time_frame]` with

shape `[4,3,600]`. Target axes are `[frequency_band,time_frame]` with shape

`[3,600]`. Band index 0 is 1–3 Hz, index 1 is 3–6 Hz, and index 2 is 6–12 Hz.

All four supports and the target use identical frame boundaries and processing.

### Submission

Submit a CSV with exactly these columns in exactly this order:

```
id,wavefield_json
```

| Column | Type | Constraint |

|---|---|---|

| `id` | string | Same ids as `test.csv` |

| `wavefield_json` | string | JSON numeric array with exact shape `[3,600]` |

`wavefield_json` must contain exactly three band lists, each containing exactly 600 finite numeric values. Every value must lie in `[0,30]`.

Example structure:

```
id,wavefield_json

wf_01ab23cd45ef67,"[[0.14,0.16,...],[0.08,0.09,...],[0.03,0.04,...]]"
```
