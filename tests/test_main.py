import pandas as pd
import pytest

from main import load_and_clean_data, train_model


@pytest.fixture
def dummy_features():
    features = pd.DataFrame(
        {
            "bill_length_mm": [50, 40, 50, 40],
            "bill_depth_mm": [15, 18, 15, 18],
            "flipper_length_mm": [200, 180, 200, 180],
            "body_mass_g": [4000, 3000, 4000, 3000],
            "island_Dream": [0, 1, 0, 1],
            "island_Torgersen": [0, 0, 0, 0],
            "sex_Male": [1, 0, 1, 0],
        }
    )
    return features


@pytest.fixture
def dummy_targets():
    targets = pd.Series(["Gentoo", "Adelie", "Gentoo", "Adelie"])
    return targets


def test_data_quality():
    """Ensure data is loaded correctly and has no missing values."""
    df = load_and_clean_data()

    # Check that we got a dataframe
    assert isinstance(df, pd.DataFrame)

    # Check that there are no empty values (NaNs) remaining
    assert df.isnull().sum().sum() == 0

    # Check that expected columns exist
    expected_cols = [
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island_Dream",
        "island_Torgersen",
        "sex_Male",
    ]
    assert set(expected_cols) == set(df.columns)


def test_model_training(dummy_features, dummy_targets):
    """Test 2: specific Check if the model can overfit a tiny dataset."""
    # Create a dummy dataset
    X_dummy = dummy_features
    y_dummy = dummy_targets

    # Train the model
    clf = train_model(X_dummy, y_dummy)

    # Predict on the same data
    preds = clf.predict(X_dummy)

    # In a tiny dataset, a Random Forest should easily get 100% accuracy
    # This proves the model logic "works"
    assert (preds == y_dummy).all()


def test_output_shapes():
    """Test 3: Ensure data splitting results in correct shapes."""
    df = load_and_clean_data()
    X = df.drop(columns=["species"])

    # We expect roughly 330+ rows in the cleaned penguins dataset
    assert X.shape[0] >= 333
    # We expect roughly 7 columns after dummy encoding
    assert X.shape[1] == 7
