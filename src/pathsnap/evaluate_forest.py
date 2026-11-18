"""Score held-out trips first, then read references in a separate evaluation phase."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from . import config, data_io
from .build import F32
from .features import build_candidate_table
from .labels import align_points_to_route
from .metrics import IDS, evaluate, score_candidates
from .network import build_geom
from .train import file_hash, write_json
from .training_handoff import convert


def prepare_trip(trip_id, partition, root):
    """Build inference inputs without opening a .route file."""
    trip = data_io.load_trip(trip_id, partition, with_route=False, root=root)
    if trip.track.ndim != 2 or trip.track.shape[1] != 3 or len(trip.track) == 0:
        raise ValueError(f"Invalid or empty GPS track: {trip_id}")
    if not np.isfinite(trip.track).all() or (np.diff(trip.track[:, 2]) <= 0).any():
        raise ValueError(f"Nonfinite GPS values or nonincreasing time: {trip_id}")
    if trip.nodes.ndim != 2 or trip.nodes.shape[1] != 2 or not np.isfinite(trip.nodes).all():
        raise ValueError(f"Invalid road nodes: {trip_id}")
    if (trip.arcs.ndim != 2 or trip.arcs.shape[1] != 2 or len(trip.arcs) == 0
            or (trip.arcs < 0).any() or (trip.arcs >= len(trip.nodes)).any()):
        raise ValueError(f"Invalid directed road arcs: {trip_id}")
    geom = build_geom(trip)
    candidates = build_candidate_table(geom)
    # Match the float32 feature representation used by training's Parquet builder.
    for name in F32:
        candidates[name] = candidates[name].astype(np.float32)
    return trip, candidates


def score_table(model, candidates, features):
    table = candidates.rename(columns={"point_idx": "observation_id", "arc_id": "candidate_id"}).copy()
    for name in IDS:
        table[name] = table[name].astype("string")
    return score_candidates(model, table, features)


def reference_partition(trip, candidates):
    """Reference-bearing objects stay in this evaluation phase only."""
    geom = build_geom(trip)
    _, arcs, distances = align_points_to_route(geom, trip.route)
    valid = distances <= config.LABEL_MAX_DIST_M
    points = pd.DataFrame({
        "trip_id": trip.trip_id, "point_idx": np.arange(len(trip.track)),
        "label_arc": arcs, "label_valid": valid,
    })
    labelled = candidates.copy()
    indices = labelled.point_idx.to_numpy(dtype=int)
    labelled["label_valid"] = valid[indices]
    labelled["label"] = (labelled.arc_id.to_numpy() == arcs[indices]).astype(int)
    return convert(labelled, points, config.FEATURE_COLUMNS)


def nearest_scores(candidates, scores):
    """Distance-only rank baseline, with the same metric/tie implementation."""
    baseline = scores[IDS].copy()
    baseline["probability"] = np.exp(-candidates.dist_m.to_numpy(dtype=float) / config.CAND_RADIUS_M)
    return baseline


def pointwise_matches(inventory, predictions):
    """Evaluate frozen decoder choices with the classifier's known-reference denominator."""
    compared = inventory.merge(predictions, on=["trip_id", "observation_id"], how="left", validate="one_to_one")
    known = compared.reference_status.eq("known")
    correct = compared.predicted_candidate_id.eq(compared.reference_candidate_id).fillna(False) & known
    return {
        "top1_accuracy": float(correct[known].mean()) if known.any() else None,
        "correct_known_observations": int(correct.sum()),
        "known_observations": int(known.sum()),
        "unresolved_observations": int((~known).sum()),
    }


def run(model_dir, root, output, partition="evaluator_test", feature_dir=None, with_hmm=False, decoder_config=None):
    model_dir, root, output = Path(model_dir), Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new, empty evaluation output directory")
    metadata = json.loads((model_dir / "run.json").read_text())
    if metadata["status"] != "complete" or metadata["model"] != "random_forest":
        raise ValueError("A completed selected Random Forest artifact is required")
    configuration = json.loads((model_dir / "configuration.json").read_text())
    if configuration["features"] != config.FEATURE_COLUMNS:
        raise ValueError("Saved model features disagree with inference feature generation")
    artifact = model_dir / "pipeline.joblib"
    model_hash = file_hash(artifact)
    if model_hash != metadata["model_sha256"]:
        raise ValueError("Model artifact hash differs from the frozen training record")
    decoder_parameters = {}
    if decoder_config:
        if not with_hmm:
            raise ValueError("Decoder configuration requires with_hmm")
        from .hmm import load_decoder_parameters
        decoder_parameters = load_decoder_parameters(decoder_config, model_sha256=model_hash)
    trips = data_io.list_trips(partition, root)
    if not trips:
        raise ValueError(f"No {partition} trips under {root}")
    # Verify separation against every original training and student-testing trip,
    # including excluded training trips; evaluator references are not needed here.
    feature_dir = Path(feature_dir or config.FEATURE_DIR)
    prior_ids = set(metadata["training_trips"]) | set(metadata["excluded_training_trips"])
    for name in ["train", "student_test"]:
        points = feature_dir / f"{name}_points.parquet"
        if file_hash(points) != metadata["input_sha256"][f"{name}_points"]:
            raise ValueError("Data used for the held-out overlap check differs from training provenance")
        prior_ids.update(pd.read_parquet(points, columns=["trip_id"]).trip_id.astype(str))
    if set(trips) & prior_ids:
        raise ValueError("Evaluator trips overlap training or student-testing trips")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "predicting", "model": "random_forest", "model_sha256": model_hash,
        "model_dir": str(model_dir.resolve()), "partition": partition, "trip_ids": trips,
        "fit_or_tuning_on_evaluator": False,
        "decoder": ({"name": "metric_hmm_viterbi", "beta": 50.0, "max_gap_s": 30.0,
                     "max_speed_mps": 60.0, "max_transition_distance_m": 5000.0,
                     "same_arc_jitter_m": 0.0, "emission_mode": "probability",
                     **decoder_parameters} if with_hmm else None),
        "pointwise_reference": "Monotone route alignment, valid if distance <= 50 m",
        "reference_limitation": "Derived reference labels, not independently supplied pointwise truth",
        "input_sha256": {}, "prediction_sha256": {},
    }
    write_json(output / "run.json", manifest)
    if decoder_config:
        configuration_record = json.loads(Path(decoder_config).read_text())
        write_json(output / "decoder_configuration.json", configuration_record)
        manifest["decoder_configuration_sha256"] = file_hash(output / "decoder_configuration.json")
    model = joblib.load(artifact)
    prepared = []
    for trip_id in trips:
        started = perf_counter()
        trip, candidates = prepare_trip(trip_id, partition, root)
        scores = score_table(model, candidates, configuration["features"])
        directory = output / "predictions" / trip_id
        directory.mkdir(parents=True)
        candidates.to_parquet(directory / "candidates.parquet", index=False)
        scores.to_csv(directory / "candidate_scores.csv", index=False)
        manifest["prediction_sha256"][trip_id] = file_hash(directory / "candidate_scores.csv")
        # Predictions are saved before opening any reference routes.
        top = scores.sort_values(IDS, kind="stable").sort_values(
            "probability", ascending=False, kind="stable").drop_duplicates(IDS[:2])
        top = top.rename(columns={"observation_id": "point_idx", "candidate_id": "predicted_arc_id"})
        top["point_idx"] = top.point_idx.astype(int)
        points = pd.DataFrame({"trip_id": trip_id, "point_idx": np.arange(len(trip.track)),
                               "t_sec": trip.track[:, 2], "lon": trip.track[:, 0], "lat": trip.track[:, 1]})
        points.merge(top, on=["trip_id", "point_idx"], how="left", validate="one_to_one").to_csv(
            directory / "observation_predictions.csv", index=False)
        if with_hmm:
            from .hmm import viterbi_decode
            geom = build_geom(trip)
            decoded = viterbi_decode(candidates, trip.arcs, scores.probability,
                arc_lengths=geom.arc_len, gps_xy=geom.P, timestamps=geom.t, return_details=True,
                **decoder_parameters)
            for segment in decoded.route_segments:
                if any(trip.arcs[first, 1] != trip.arcs[second, 0]
                       for first, second in zip(segment, segment[1:])):
                    raise RuntimeError("Reconstructed route contains a disconnected segment")
            matches = points.copy()
            matches["predicted_arc_id"] = pd.array(decoded.point_arc_ids, dtype="Int64")
            matches.loc[matches.predicted_arc_id.eq(-1), "predicted_arc_id"] = pd.NA
            matches["segment_id"] = decoded.segment_ids
            matches.to_csv(directory / "hmm_observation_predictions.csv", index=False)
            write_json(directory / "hmm_route.json", asdict(decoded))
        seconds = perf_counter() - started
        prepared.append((trip_id, seconds))
        for extension in ["track", "nodes", "arcs"]:
            path = root / partition / trip_id / f"{trip_id}.{extension}"
            manifest["input_sha256"][f"{trip_id}.{extension}"] = file_hash(path)
        print(f"Predicted {trip_id}: {len(trip.track):,} points, {len(scores):,} candidates, {seconds:.1f}s", flush=True)
    manifest["status"] = "evaluating_frozen_predictions"
    write_json(output / "run.json", manifest)
    partitions, score_tables, baseline_tables, rows, hmm_tables = [], [], [], [], []
    for trip_id, seconds in prepared:
        for extension in ["track", "nodes", "arcs"]:
            path = root / partition / trip_id / f"{trip_id}.{extension}"
            if file_hash(path) != manifest["input_sha256"][f"{trip_id}.{extension}"]:
                raise RuntimeError("Inference inputs changed after predictions were saved")
        trip = data_io.load_trip(trip_id, partition, with_route=True, root=root)
        if trip.route is None or len(trip.route) == 0 or (trip.route < 0).any() or (trip.route >= len(trip.arcs)).any():
            raise ValueError(f"Missing or invalid evaluation reference route: {trip_id}")
        reference = root / partition / trip_id / f"{trip_id}.route"
        manifest["input_sha256"][f"{trip_id}.route"] = file_hash(reference)
        directory = output / "predictions" / trip_id
        if file_hash(directory / "candidate_scores.csv") != manifest["prediction_sha256"][trip_id]:
            raise RuntimeError("Frozen candidate predictions changed before evaluation")
        candidates = pd.read_parquet(directory / "candidates.parquet")
        scores = pd.read_csv(directory / "candidate_scores.csv", dtype={key: "string" for key in IDS})
        part = reference_partition(trip, candidates)
        metrics, observations = evaluate(part.candidates, part.inventory, scores)
        baseline = nearest_scores(candidates, scores)
        baseline_metrics, _ = evaluate(part.candidates, part.inventory, baseline)
        metrics.update({"trip_id": trip_id, "observations": len(trip.track),
                        "candidate_rows": len(candidates), "prediction_seconds": seconds,
                        "nearest_top1_accuracy": baseline_metrics["top1_accuracy"],
                        "reference_coverage": metrics["known_observations"] / len(trip.track)})
        if with_hmm:
            predicted = pd.read_csv(directory / "hmm_observation_predictions.csv",
                dtype={"trip_id": "string", "point_idx": "string", "predicted_arc_id": "string"})
            predicted = predicted.rename(columns={"point_idx": "observation_id", "predicted_arc_id": "predicted_candidate_id"})
            predicted = predicted[["trip_id", "observation_id", "predicted_candidate_id"]]
            hmm_metrics = pointwise_matches(part.inventory, predicted)
            metrics["hmm_top1_accuracy"] = hmm_metrics["top1_accuracy"]
            metrics["hmm_correct_known_observations"] = hmm_metrics["correct_known_observations"]
            hmm_tables.append(predicted)
        directory = output / "evaluation" / trip_id
        directory.mkdir(parents=True)
        observations.to_csv(directory / "reference_comparison.csv", index=False)
        write_json(directory / "metrics.json", metrics)
        partitions.append(part)
        score_tables.append(scores)
        baseline_tables.append(baseline)
        rows.append(metrics)
        accuracy = metrics["top1_accuracy"]
        print(f"Evaluated {trip_id}: Top-1={accuracy:.2%}" if accuracy is not None else
              f"Evaluated {trip_id}: no valid reference observations", flush=True)
    candidates = pd.concat([part.candidates for part in partitions], ignore_index=True)
    inventory = pd.concat([part.inventory for part in partitions], ignore_index=True)
    scores = pd.concat(score_tables, ignore_index=True)
    metrics, observations = evaluate(candidates, inventory, scores)
    baseline_metrics, _ = evaluate(candidates, inventory, pd.concat(baseline_tables, ignore_index=True))
    table = pd.DataFrame(rows)
    metrics.update({
        "trips": len(trips), "observations": len(inventory), "candidate_rows": len(candidates),
        "correct_known_observations": int(observations.correct.sum()),
        "reference_coverage": metrics["known_observations"] / len(inventory),
        "mean_trip_top1_accuracy": float(table.top1_accuracy.mean()) if table.top1_accuracy.notna().any() else None,
        "nearest_top1_accuracy": baseline_metrics["top1_accuracy"],
        "prediction_seconds": float(table.prediction_seconds.sum()),
        "note": "Pooled observation Top-1 against valid derived references; no HMM/Viterbi",
    })
    if with_hmm:
        hmm_metrics = pointwise_matches(inventory, pd.concat(hmm_tables, ignore_index=True))
        metrics.update({"hmm_top1_accuracy": hmm_metrics["top1_accuracy"],
                        "hmm_correct_known_observations": hmm_metrics["correct_known_observations"],
                        "mean_trip_hmm_top1_accuracy": float(table.hmm_top1_accuracy.mean()),
                        "note": "Classifier and RF+HMM pointwise accuracy on identical valid derived references"})
    table.to_csv(output / "per_trip_metrics.csv", index=False)
    write_json(output / "metrics.json", metrics)
    if file_hash(artifact) != model_hash:
        raise RuntimeError("Model artifact changed during evaluation")
    manifest["status"] = "complete"
    write_json(output / "run.json", manifest)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-root", required=True, help="Parent directory containing evaluator_test/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-dir", default=str(config.FEATURE_DIR))
    parser.add_argument("--with-hmm", action="store_true")
    parser.add_argument("--decoder-config")
    args = parser.parse_args()
    run(args.model_dir, args.data_root, args.output, feature_dir=args.feature_dir, with_hmm=args.with_hmm,
        decoder_config=args.decoder_config)


if __name__ == "__main__":
    main()
