# Viva notes - Person 1 (say it in plain words)

**One-line summary.** "For every GPS point I list the nearby road segments, describe each (point, segment) pair with numbers, and later models score which segment is right."

**Q: Why not predict the road directly with a classifier over all arcs?**  A trip has up to 300k arcs. Radius-based candidates shrink it to ~16 per point; cost = Candidate Miss Rate (0.87% at 75 m).

**Q: Why UTM?** Lon/lat are angles; 1 deg lon is shorter than 1 deg lat and changes with latitude. UTM gives metres with <0.04% error inside a zone. Per-trip zone because trips are in 25 zones.

**Q: Why is .route not enough as a label?** It lists ~280 arcs for ~1500 points. I align points to the route with a monotone DP: j1<=j2<=...<=jN minimising summed point-arc distance (capped at 150 m). Runs in O(N*M) with a running prefix-min.

**Q: Which trips did you drop and why?** 00000033 and 00000054: the reference covers 13%/19% of the GPS path, so labels are wrong for most points. Also points >50 m from reference are not used as labels. Both rules are in `config.py`, measured, and documented.

**Q: Is the radius study circular?** For *valid-label* points the miss rate is 0 by construction (valid = within 50 m). So I report the miss rate over ALL points of reliable trips (0.87% at 75 m).

**Q: How do you prevent leakage?** Feature code never opens `.route` (test `test_features_identical_without_route_file`); folds are by trip; no learned preprocessing; evaluator data never present; evaluator labels stay server-side.

**Q: Why two arcs at the same distance, and how do you separate them?** Two-way roads have a->b and b->a. Heading difference, signed progress along the arc across i-1 -> i+1, and `has_reverse` break the tie.

**Q: What is the context window?** For candidate arc e at point i, distances from e to points i-3..i+3 (ignoring neighbours >10 s away). A wrong parallel street fits the neighbours badly.

**Q: Weakest points / limitations?** Labels are derived (not given); 75 m radius misses true arcs when GPS jumps far; features are label-free so they are safe at inference but cannot learn road type (not in data).

**Numbers to remember:** 80/10 trips, 208k/19k points, 96.7% 1-Hz steps, 92% two-way arcs, 75 m radius, 15.8 cand/pt, 0.87% miss, 7% positive rows, 22 features, 5 folds, 2 unreliable trips.
