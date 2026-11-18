import ast
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from pathsnap import train
from pathsnap.kaggle_bundle import create_bundle, import_results
from test_training import write_inputs


def test_baseline_only_resume_and_import(tmp_path, monkeypatch):
    write_inputs(tmp_path)
    remote = tmp_path / "kaggle_run"
    args = train.parser().parse_args([
        "--train", str(tmp_path / "train.csv"), "--student-test", str(tmp_path / "test.csv"),
        "--train-inventory", str(tmp_path / "train_inventory.csv"),
        "--student-test-inventory", str(tmp_path / "test_inventory.csv"),
        "--features", str(tmp_path / "features.json"), "--output", str(remote),
        "--models", "adaboost", "--baseline-only", "--allow-partial-data",
    ])
    comparison = train.run(args)
    assert comparison.model.tolist() == ["adaboost"]
    assert not (remote / "selection.json").exists()
    assert not (remote / "optimisation.sqlite3").exists()
    args.resume = True
    def no_retraining(*args, **kwargs):
        raise AssertionError("Resume must reuse complete baseline results")
    monkeypatch.setattr(train, "cross_validate", no_retraining)
    monkeypatch.setattr(train, "refit_and_evaluate", no_retraining)
    train.run(args)
    local = tmp_path / "local"
    local.mkdir()
    for filename in ["run.json", "fold_assignments.csv"]:
        shutil.copy2(remote / filename, local / filename)
    imported = import_results(remote, local)
    assert (imported / "pipeline.joblib").exists()
    assert (imported / "execution.json").exists()
    with pytest.raises(ValueError, match="not empty"):
        import_results(remote, local)
    metadata = json.loads((local / "run.json").read_text())
    metadata["seed"] = 99
    (local / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="seed differs"):
        import_results(remote, local)
    candidates = pd.read_csv(tmp_path / "train.csv")
    candidates.loc[0, "distance"] += 1
    candidates.to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match="differ from the original"):
        train.run(args)


def test_bundle_layout_and_notebook_syntax(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    for filename in ["feature_schema.json", "train_candidates.parquet", "train_points.parquet",
                     "student_test_candidates.parquet", "student_test_points.parquet"]:
        (features / filename).write_text("synthetic bundle-layout fixture")
    output = create_bundle(features, tmp_path / "upload.zip")
    with ZipFile(output) as archive:
        names = archive.namelist()
        assert "features/train_candidates.parquet" in names
        assert "src/pathsnap/train.py" in names
        assert "requirements-kaggle.txt" in names
        assert not any(".route" in name or "data/raw" in name for name in names)
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/adaboost_kaggle.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "adaboost_kaggle.ipynb", "exec")


@pytest.mark.parametrize("extracted", [True, False])
def test_notebook_accepts_kaggle_extracted_or_zipped_input(tmp_path, extracted):
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/adaboost_kaggle.ipynb").read_text())
    parsed = ast.parse("".join(notebook["cells"][1]["source"]))
    definitions = ast.Module(body=[node for node in parsed.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[])
    namespace = {}
    exec(compile(definitions, "notebook setup", "exec"), namespace)
    input_dir = tmp_path / "input" / "adaboost"
    input_dir.mkdir(parents=True)
    files = {"requirements-kaggle.txt": "test requirements", "features/feature_schema.json": "{}",
             "src/pathsnap/train.py": "# synthetic fixture"}
    if extracted:
        for name, content in files.items():
            path = input_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    else:
        with ZipFile(input_dir / "pathsnap_adaboost.zip", "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
    result = namespace["prepare_bundle"](tmp_path / "input", tmp_path / "working")
    assert (result / "features/feature_schema.json").read_text() == "{}"
    assert (result / "src/pathsnap/train.py").is_file()
