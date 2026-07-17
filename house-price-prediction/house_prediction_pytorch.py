import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CITY_MAPPING = {
    "Chennai": 1,
    "Coimbatore": 2,
}


def load_data(file_path="house_data_1500.csv"):
    return pd.read_csv(file_path)


def encode_city(df):
    df = df.copy()
    df["city_tier"] = df["city"].map(CITY_MAPPING)
    df.drop("city", axis=1, inplace=True)
    return df


def prepare_features(df):
    x = df[
        [
            "sqft",
            "bedrooms",
            "toilets",
            "parking",
            "year",
            "city_tier",
        ]
    ]

    y = df["price_lakhs"]

    return x, y


def split_and_scale_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return (
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        scaler,
    )


def convert_to_tensors(
    x_train_scaled,
    x_test_scaled,
    y_train,
    y_test,
):
    x_train_tensor = torch.tensor(
        x_train_scaled,
        dtype=torch.float32,
    )

    x_test_tensor = torch.tensor(
        x_test_scaled,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32,
    ).view(-1, 1)

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32,
    ).view(-1, 1)

    return (
        x_train_tensor,
        x_test_tensor,
        y_train_tensor,
        y_test_tensor,
    )


class HousePriceNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def build_model():
    return HousePriceNN()


def train_model(
    model,
    x_train_tensor,
    y_train_tensor,
    epochs=5,
):
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.MSELoss()

    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()

        outputs = model(x_train_tensor)

        loss = criterion(
            outputs,
            y_train_tensor,
        )

        loss.backward()

        optimizer.step()

    return model


def predict_house(
    model,
    scaler,
    sqft,
    bedrooms,
    toilets,
    parking,
    year,
    city_tier,
):
    sample = np.array(
        [
            [
                sqft,
                bedrooms,
                toilets,
                parking,
                year,
                city_tier,
            ]
        ]
    )

    sample_scaled = scaler.transform(sample)

    sample_tensor = torch.tensor(
        sample_scaled,
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        prediction = model(sample_tensor)

    return float(prediction.item())


def main():
    df = load_data()

    df = encode_city(df)

    x, y = prepare_features(df)

    (
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        scaler,
    ) = split_and_scale_data(x, y)

    (
        x_train_tensor,
        x_test_tensor,
        y_train_tensor,
        y_test_tensor,
    ) = convert_to_tensors(
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
    )

    model = build_model()

    train_model(
        model,
        x_train_tensor,
        y_train_tensor,
    )

    price = predict_house(
        model,
        scaler,
        1500,
        3,
        3,
        1,
        2030,
        1,
    )

    print(f"Predicted Chennai Price: {price:.2f} Lakhs")


if __name__ == "__main__":
    main()
