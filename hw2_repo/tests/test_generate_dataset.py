from hw2.dataset import DataConfig, generate_dataset


def test_generate_dataset():
    cfg = DataConfig(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=1,
        test_size=0.2
    )
    x_train, x_test, y_train, y_test = generate_dataset(cfg)

    assert not x_train.empty and not x_test.empty
    assert len(x_train) + len(x_test) == cfg.n_samples
    assert set(y_train[0].unique()).issubset({0, 1})
