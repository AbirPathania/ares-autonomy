# Phase 0, the LiDAR ablation, synthetic

What accuracy the traversability estimate loses when the LiDAR is
removed and stereo depth is put in its place. Identical estimator,
identical frames, identical vehicle limits. Only the depth source
changes.

Hull revision H-2. Slope limit 26.6 degrees, derived from a centre of mass at 209 mm and the track, divided by the same safety factor the stopping distance uses.

## Headline

| Scene | LiDAR accuracy | Stereo accuracy | Accuracy lost | False free space, LiDAR | stereo | added |
|---|---|---|---|---|---|---|
| `flat_textured` | 1.000 | 1.000 | **+0.000** | n/a | n/a | **n/a** |
| `step_field` | 0.865 | 0.675 | **+0.190** | 0.0000 | 0.0000 | **+0.0000** |
| `trench_field` | 0.940 | 0.915 | **+0.025** | 0.0622 | 0.1552 | **+0.0931** |
| `snow_scree` | 0.826 | 0.924 | **-0.097** | n/a | n/a | **n/a** |

Accuracy is computed over cells the estimator was willing to decide.
False free space is a true obstacle called traversable, counted against
every true obstacle cell including the undecided ones, because declining
to decide is safe and being wrong is not. It is reported separately and
never averaged into accuracy, which would hide it behind a large easy
background.

`n/a` means the scene contains no obstacle cells, so the metric has no
denominator. On such a scene accuracy reduces to recall on traversable
ground, and a stereo run that scores above the LiDAR run has not beaten
it at anything that matters.

## `flat_textured`

Mean stereo confidence 0.412. 42.3 per cent of pixels cleared the trust threshold. 85.0 per cent of grid cells came back unknown from stereo, against 91.7 from LiDAR.

Measured stereo depth error against the section 03.2 prediction:

| Range | Samples | Bias | Measured sigma | Predicted | Ratio |
|---|---|---|---|---|---|
| 1 to 2 m | 55711 | +0.0066 m | 0.0082 m | 0.0084 m | 0.97 |
| 2 to 3 m | 24104 | +0.0139 m | 0.0215 m | 0.0231 m | 0.93 |
| 3 to 5 m | 19243 | +0.0360 m | 0.0458 m | 0.0570 m | 0.80 |
| 5 to 7 m | 8490 | +0.0686 m | 0.1089 m | 0.1355 m | 0.80 |
| 7 to 10 m | 3789 | +0.0734 m | 0.1609 m | 0.2405 m | 0.67 |

A ratio near or below one means the matcher is achieving the subpixel disparity precision the tables assume. Above one means it is not, and the range at which the depth is trustworthy is shorter than section 03.2 claims.

Traversability by range, stereo against LiDAR:

| Range | LiDAR accuracy | Stereo accuracy | Stereo unknown |
|---|---|---|---|
| 0-2 m | 1.000 | 1.000 | 0.553 |
| 2-4 m | 1.000 | 1.000 | 0.696 |
| 4-6 m | 1.000 | 1.000 | 0.735 |
| 6-8 m | 1.000 | 0.997 | 0.899 |
| 8-12 m | 1.000 | n/a | 1.000 |
| 12-20 m | n/a | n/a | 1.000 |

## `step_field`

Mean stereo confidence 0.407. 40.8 per cent of pixels cleared the trust threshold. 86.0 per cent of grid cells came back unknown from stereo, against 90.8 from LiDAR.

Measured stereo depth error against the section 03.2 prediction:

| Range | Samples | Bias | Measured sigma | Predicted | Ratio |
|---|---|---|---|---|---|
| 1 to 2 m | 52861 | +0.0068 m | 0.0087 m | 0.0086 m | 1.01 |
| 2 to 3 m | 24099 | +0.0148 m | 0.0216 m | 0.0231 m | 0.94 |
| 3 to 5 m | 24863 | +0.0164 m | 0.0503 m | 0.0582 m | 0.86 |
| 5 to 7 m | 6272 | +0.0793 m | 0.1164 m | 0.1355 m | 0.86 |
| 7 to 10 m | 6687 | +0.1619 m | 0.1744 m | 0.2300 m | 0.76 |

A ratio near or below one means the matcher is achieving the subpixel disparity precision the tables assume. Above one means it is not, and the range at which the depth is trustworthy is shorter than section 03.2 claims.

Traversability by range, stereo against LiDAR:

| Range | LiDAR accuracy | Stereo accuracy | Stereo unknown |
|---|---|---|---|
| 0-2 m | 1.000 | 1.000 | 0.603 |
| 2-4 m | 0.985 | 0.971 | 0.696 |
| 4-6 m | 0.438 | 0.324 | 0.752 |
| 6-8 m | 0.671 | 0.369 | 0.920 |
| 8-12 m | 0.486 | 0.500 | 1.000 |
| 12-20 m | n/a | n/a | 1.000 |

## `trench_field`

Mean stereo confidence 0.389. 39.9 per cent of pixels cleared the trust threshold. 85.7 per cent of grid cells came back unknown from stereo, against 91.9 from LiDAR.

Measured stereo depth error against the section 03.2 prediction:

| Range | Samples | Bias | Measured sigma | Predicted | Ratio |
|---|---|---|---|---|---|
| 1 to 2 m | 53835 | +0.0075 m | 0.0088 m | 0.0085 m | 1.04 |
| 2 to 3 m | 24073 | +0.0127 m | 0.0219 m | 0.0231 m | 0.95 |
| 3 to 5 m | 18562 | +0.0324 m | 0.0869 m | 0.0605 m | 1.44 |
| 5 to 7 m | 8493 | +0.0626 m | 0.1148 m | 0.1355 m | 0.85 |
| 7 to 10 m | 3901 | +0.0826 m | 0.1645 m | 0.2412 m | 0.68 |

A ratio near or below one means the matcher is achieving the subpixel disparity precision the tables assume. Above one means it is not, and the range at which the depth is trustworthy is shorter than section 03.2 claims.

Traversability by range, stereo against LiDAR:

| Range | LiDAR accuracy | Stereo accuracy | Stereo unknown |
|---|---|---|---|
| 0-2 m | 1.000 | 1.000 | 0.578 |
| 2-4 m | 0.905 | 0.808 | 0.723 |
| 4-6 m | 0.914 | 0.955 | 0.741 |
| 6-8 m | 1.000 | 1.000 | 0.898 |
| 8-12 m | 1.000 | n/a | 1.000 |
| 12-20 m | n/a | n/a | 1.000 |

## `snow_scree`

Mean stereo confidence 0.380. 41.2 per cent of pixels cleared the trust threshold. 85.1 per cent of grid cells came back unknown from stereo, against 93.0 from LiDAR.

Measured stereo depth error against the section 03.2 prediction:

| Range | Samples | Bias | Measured sigma | Predicted | Ratio |
|---|---|---|---|---|---|
| 1 to 2 m | 55737 | +0.0112 m | 0.0198 m | 0.0083 m | 2.39 |
| 2 to 3 m | 23854 | +0.0176 m | 0.0456 m | 0.0226 m | 2.02 |
| 3 to 5 m | 18739 | +0.0272 m | 0.0916 m | 0.0563 m | 1.63 |
| 5 to 7 m | 5438 | -0.0884 m | 0.2354 m | 0.1230 m | 1.91 |

A ratio near or below one means the matcher is achieving the subpixel disparity precision the tables assume. Above one means it is not, and the range at which the depth is trustworthy is shorter than section 03.2 claims.

Traversability by range, stereo against LiDAR:

| Range | LiDAR accuracy | Stereo accuracy | Stereo unknown |
|---|---|---|---|
| 0-2 m | 0.783 | 0.865 | 0.542 |
| 2-4 m | 0.815 | 0.950 | 0.640 |
| 4-6 m | 0.960 | 0.954 | 0.739 |
| 6-8 m | 1.000 | 0.630 | 0.960 |
| 8-12 m | 1.000 | n/a | 1.000 |
| 12-20 m | n/a | n/a | 1.000 |

