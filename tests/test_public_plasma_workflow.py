from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_readme_and_console_scripts_expose_only_modern_euv_workflows():
    readme = (ROOT / "README.md").read_text()
    project = (ROOT / "pyproject.toml").read_text()

    assert "sunerf-responses" in readme
    assert "python -m sunerf.data.euv.prepare" in readme
    assert "python -m sunerf.run_plasma" in readme
    assert "sunerf.run_emission" not in readme
    assert "sunerf.data.prep.sdo" not in readme
    assert "sunerf.data.prep.stereo" not in readme
    assert "config/emission/" not in readme

    for name in ("sunerf-responses", "sunerf-prepare-euv", "sunerf-plasma"):
        assert name in project
    assert "run_emission:main" not in project


def test_removed_legacy_modules_and_scripts_are_absent():
    removed = (
        ROOT / "sunerf" / "data" / "euv" / "load_aia_response_function.py",
        ROOT / "sunerf" / "data" / "euv" / "prep_aia.py",
        ROOT / "sunerf" / "model" / "emission.py",
        ROOT / "sunerf" / "run_emission.py",
        ROOT / "scripts" / "learn_response_functions.sh",
        ROOT / "config" / "emission",
    )
    assert all(not path.exists() for path in removed)
