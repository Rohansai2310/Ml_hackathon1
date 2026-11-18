# Draft for Report Section 2.2 - Dataset Understanding and EDA
*(Numbers come from `outputs/eda/*.csv`. Rewrite in your own words before submitting; keep the numbers. Disclose AI assistance as the event policy requires.)*

## How did you preprocess the data?
Partition used: **train (80 trips)** for all fitting-related analysis, **student_test (10 trips)** for development-stage checks; evaluator-test never read.

1. **Parsing.** `.track` = lon, lat, t(s); `.nodes` = lon, lat (row = node id); `.arcs` = from_node, to_node (row = arc id); `.route` = ordered reference arc ids. All 90 trips have all four files; no NaN, no (0,0) or out-of-range coordinates, no duplicate or backward timestamps (`missing_value_analysis.csv`).
2. **Projection to metres.** Degrees are angles, so distances in degrees are wrong. Each trip is projected to the UTM zone of its own centroid (trips span 25 UTM zones, lon -100 to +137). All distances below are in metres.
3. **Candidate generation.** Each trip's network has up to 307k arcs, so for every GPS point we keep arcs within **75 m**, at most the 40 nearest. Radius study (train, reliable trips): miss rate 3.2% @25 m, 1.4% @50 m, **0.87% @75 m**, 0.60% @100 m, 0.42% @150 m, while candidates/point grow 4.8 -> 9.6 -> **15.8** -> 23.3 -> 42.6. 75 m is the knee: most of the gain, without a 50% bigger table. The remaining misses are mostly noisy labels, not geometry.
4. **Labels.** `.route` has ~280 arcs for ~1500 points, so it is not one label per point. Each point is aligned to a route arc with a monotone DP (the vehicle can only move forward along the reference route). Candidate row label = 1 iff its arc is the aligned arc. 98.6% of points in reliable train trips are within 50 m of their aligned arc (median ~5 m) = `label_valid`.
5. **Reference limitation handling.** Trips 00000033 and 00000054: the reference route is only 13% / 19% of the GPS path length and 59% / 89% of their points are >50 m from it -> flagged unreliable and **excluded from training/CV** (78 trips, 183,260 points used). Other points >50 m (1.27%) are dropped as labels only.
6. **CV folds.** 5 group-aware folds by trip (greedy balanced by point count) -> no trip's points appear on both sides of a split (`folds.csv`).
7. **No learned preprocessing** (features are deterministic geometry), so nothing is fitted on student/evaluator data.

## What patterns did you observe?
- **Partition sizes** (`dataset_inventory.csv`): train 80 trips / 208,417 points; student_test 10 trips / 19,160 points. Median trip ~1,500 points, range 262-19,313.
- **Sampling** (fig 02): 96.7% of steps are exactly 1 s; 1,260 gaps >5 s, 135 gaps >30 s; trips 00000090 and 00000078 contain recording pauses of 153,064 s and 52,810 s.
- **GPS noise** (fig 03, 06): 555 steps in 20 trips imply >60 m/s (GPS jumps); GPS-to-reference distance median ~5 m but with a long tail.
- **Road networks** (fig 04, `trip_summary.csv`): arcs ~ 2 x nodes because 92% of arcs have a reverse arc (two-way roads); 89 of 90 networks are not fully connected (median 19 weakly-connected components, giant component ~98% of nodes); no isolated or invalid nodes; 24 self-loops and 106 zero-length arcs.
- **Reference route** (fig 07): 14 breaks (end of arc i != start of arc i+1) in 11 trips; 10 trips revisit arcs. Treated as *reference*, not perfect truth.
- **Class imbalance:** in a 600k-row train sample only ~7% of candidate rows are the correct arc -> use Macro-F1/Top-1/log-loss, not plain accuracy.

## Which features and why? (fig 11 = single-feature strength, fig 12 = correlation)
| Group | Features | Why |
|---|---|---|
| Distance | `dist_m`, `dist_rank`, `dist_rel_min`, `n_cand` | nearest road is usually right (strongest single group, |AUC-0.5|x2 up to 0.91) |
| Position | `frac_along`, `arc_len_m`, `to/from_node_dist_m` | long arcs are likelier; near-junction points are ambiguous |
| Direction | `heading_diff_deg`, `cos_heading`, `along_advance_m`, `has_reverse` | the two directions of a two-way road are equally near; only heading/progress separates them (nearest-arc alone is right only ~57% on student_test) |
| Motion | `speed_mps`, `dt_prev`, `dt_next` | reliability of heading; gap awareness |
| Context | `win_mean/min/max_dist`, `win_hit_frac` (+-3 neighbours) | a noisy point may be nearer a parallel street, but its neighbours vote for the true road |
| Topology | `out_deg_to`, `in_deg_from`, `speed_dist_resid` | junction structure; consistency of arc direction with motion |

Honest note: `dt_prev`, `dt_next`, `frac_along`, `in/out_deg` are weak alone (AUC ~0.5); kept because they may matter in interactions - Person 2 can ablate them.
Sanity check (not a final model): a quick LightGBM on these features reaches ~86% observation Top-1 on student_test vs ~57% for nearest-arc.

## Figures / tables to paste
`01_points_per_trip`, `02_sampling_interval`, `03_speed`, `04_network_size`, `05_geographic_overview`, `06_label_distance`, `07_reference_reliability`, `08_data_quality_issues`, `09_trajectory_examples`, `10_radius_study`, `11_feature_strength`, `12_feature_correlation`; Folium screenshot of `outputs/maps/train_00000048.html` (state: partition = train).
