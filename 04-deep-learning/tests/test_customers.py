from customers import load_customers, split_frame, compare_models


def test_split_matches_day1_sizes_and_leaves_the_csv_in_place():
    frame = load_customers()
    train, validation, test = split_frame(frame)
    assert len(frame) == 2400
    assert (len(train), len(validation), len(test)) == (1440, 480, 480)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)


def test_mlp_validation_score_beats_the_forest_by_a_little():
    train, validation, _test = split_frame(load_customers())
    result = compare_models(train, validation, (32,))
    again = compare_models(train, validation, (32,))
    assert round(result["classic_validation_roc_auc"], 3) == 0.776
    assert round(result["mlp_validation_roc_auc"], 3) == 0.782
    assert round(result["classic_train_roc_auc"], 3) == 0.959
    assert round(result["mlp_train_roc_auc"], 3) == 0.849
    assert result["mlp_validation_roc_auc"] > result["classic_validation_roc_auc"]
    assert result["mlp"].named_steps["model"].n_iter_ == 200
    assert again["mlp_validation_roc_auc"] == result["mlp_validation_roc_auc"]
    assert again["classic_validation_roc_auc"] == result["classic_validation_roc_auc"]
