from house_prediction_pytorch import (
    CITY_MAPPING,
    load_data,
    encode_city,
    prepare_features,
    build_model,
)


def test_dataset_loaded():
    df = load_data()

    assert len(df) > 0


def test_required_columns_exist():
    df = load_data()

    expected_columns = {
        "sqft",
        "bedrooms",
        "toilets",
        "parking",
        "year",
        "city",
        "price_lakhs",
    }

    assert expected_columns.issubset(set(df.columns))


def test_city_mapping():
    assert CITY_MAPPING["Chennai"] == 1
    assert CITY_MAPPING["Coimbatore"] == 2


def test_city_encoding():
    df = load_data()

    encoded_df = encode_city(df)

    assert "city_tier" in encoded_df.columns
    assert "city" not in encoded_df.columns


def test_feature_count():
    df = load_data()

    encoded_df = encode_city(df)

    x, y = prepare_features(encoded_df)

    assert x.shape[1] == 6


def test_model_creation():
    model = build_model()

    assert model is not None
