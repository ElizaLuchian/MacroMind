from src.experiment_smallset import run_small_dataset_experiment


def test_small_experiment_runs_with_tfidf_backend():
    result = run_small_dataset_experiment(embedder_kwargs={"backend": "tfidf", "random_state": 0})
    assert "simulation" in result and "metrics" in result
    sim = result["simulation"]
    metrics = result["metrics"]
    assert len(sim.prices) > 0
    assert "directional_accuracy" in metrics

