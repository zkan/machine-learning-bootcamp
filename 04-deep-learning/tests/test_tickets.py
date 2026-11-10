from tickets import (
    CATEGORIES,
    PER_CLASS,
    check_tickets,
    fit_text,
    generate_tickets,
    load_tickets,
    split_tickets,
)

CUP = (
    ((16,), 0.001, 0.508),
    ((32,), 0.001, 0.700),
    ((32, 16), 0.001, 0.700),
    ((32,), 0.01, 0.758),
)


def test_ticket_file_matches_the_generator():
    check_tickets()
    frame = load_tickets()
    again = generate_tickets()
    assert frame.equals(again)
    assert len(frame) == PER_CLASS * len(CATEGORIES)
    assert set(frame["category"]) == set(CATEGORIES)
    assert frame["category"].value_counts().eq(PER_CLASS).all()
    assert frame["text"].str.len().gt(0).all()


def test_cup_accuracies_on_the_starter_configs():
    train, validation = split_tickets(load_tickets())
    assert (len(train), len(validation)) == (360, 120)
    assert validation["category"].value_counts().eq(30).all()
    for hidden, rate, expected in CUP:
        fitted = fit_text(
            train,
            validation,
            hidden_layer_sizes=hidden,
            learning_rate_init=rate,
        )
        assert round(fitted["validation_accuracy"], 3) == expected
        assert fitted["training_seconds"] > 0
        assert fitted["pipeline"].named_steps["model"].n_iter_ == 40
