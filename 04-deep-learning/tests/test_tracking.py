import mlflow

from customers import fit_timed, load_customers, mlp_pipeline, split_frame
from tickets import confusion_frame, load_tickets, split_tickets
from tracking import leaderboard, log_confusion_csv, log_cup_run, register_pipeline


def test_leaderboard_puts_the_higher_score_first(tmp_path):
    log_cup_run(
        tmp_path,
        {
            "hidden_layer_sizes": (16,),
            "learning_rate_init": 0.001,
            "validation_accuracy": 0.508,
            "training_seconds": 0.02,
        },
    )
    log_cup_run(
        tmp_path,
        {
            "hidden_layer_sizes": (32,),
            "learning_rate_init": 0.01,
            "validation_accuracy": 0.758,
            "training_seconds": 0.03,
        },
    )
    board = leaderboard(tmp_path)
    assert list(board["validation_accuracy"]) == [0.758, 0.508]
    assert board.iloc[0]["hidden_layer_sizes"] == "(32,)"
    assert (tmp_path / "tracking").is_dir()


def test_confusion_artifact_and_registry_version(tmp_path):
    run_id = log_cup_run(
        tmp_path,
        {
            "hidden_layer_sizes": (32,),
            "learning_rate_init": 0.01,
            "validation_accuracy": 0.758,
            "training_seconds": 0.03,
        },
    )
    _train, validation = split_tickets(load_tickets())
    matrix = confusion_frame(validation, validation["category"])
    log_confusion_csv(tmp_path, run_id, matrix, tmp_path / "tracking" / "confusion.csv")
    names = [item.path for item in mlflow.MlflowClient().list_artifacts(run_id)]
    assert "confusion.csv" in names

    customers = load_customers()
    train, _validation, _test = split_frame(customers)
    pipeline = mlp_pipeline(hidden_layer_sizes=(8,), max_iter=5)
    fit_timed(pipeline, train.sample(60, random_state=42))
    version = register_pipeline(tmp_path, pipeline, "mlp")
    assert version == "1"
