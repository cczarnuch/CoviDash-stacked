import os
import json
import argparse
import collections
import multiprocessing
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import preprocess as pp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.device = device

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def create_scaler():
    return MinMaxScaler(feature_range=(-1, 1))


def normalize_data(to_normalize, scaler):
    """Normalizes data with the global scaler on a scale of -1 to 1.

    Args:
        to_normalize (numpy.ndarray): Numpy array of values to be normalized.
        reset_scaler (bool, optional): If true, reset the scaler. Defaults to False.

    Returns:
        numpy.ndarray: Normalized data.
    """
    normalized_data = scaler.fit_transform(to_normalize.reshape(-1, 1))
    return normalized_data


def denormalize_data(to_denormalize, scaler):
    """Denormalizes previously normalized data using the global scaler.

    Args:
        to_denormalize (numpy.ndarray): Numpy array containing normalized values.

    Returns:
        numpy.ndarray: Denormalized data.
    """
    denormalized_data = scaler.inverse_transform(
        np.array(to_denormalize.reshape(-1, 1))
    )
    return denormalized_data


def create_tensors(normalized_train_data, train_window):
    """Creates tensors that are used for training the LSTM.

    Args:
        normalized_train_data (numpy.ndarray): Numpy array of normalized training data.
        train_window (int): Window to be used for training.

    Returns:
        [torch.Tensor]: in out sequence used for training the LSTM.
    """
    normalized_train_data = torch.FloatTensor(normalized_train_data).view(-1)

    inout_seq = []
    L = len(normalized_train_data)
    for i in range(L - train_window):
        train_seq = normalized_train_data[i: i + train_window]
        train_label = normalized_train_data[i +
                                            train_window: i + train_window + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def train_model(inout_seq):
    """Trains the LSTM model.

    Args:
        inout_seq ([torch.Tensor]): In out sequence of tensors.

    Returns:
        object: LSTM model.
    """
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

    for i in range(epochs):
        for seq, labels in inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 0:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    return model


def predict(model, num_pred, train_data_normalized, train_window):
    """Makes predictions using LSTM model and normalized training data.

    Args:
        model (object): LSTM model.
        num_pred (int): Number of predictions to make.
        train_data_normalized (numpy.ndarray): Normalized data used to train the LSTM.
        train_window (int): Moving window used to aid in predictions.

    Returns:
        numpy.ndarray: Numpy array of predictions.
    """
    test_inputs = train_data_normalized[-train_window:]
    model.eval()
    for _ in range(num_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            test_inputs = np.append(test_inputs, model(seq).item())
    return test_inputs[-num_pred:]


def convert_json(save_location, output_dict, prediction_dates=[]):
    new_dict = collections.defaultdict(list)
    for location, data in output_dict.items():
        for datenum, date in enumerate(data["dates"]):
            is_pred = True if date in prediction_dates else False
            new_dict[date].append(
                {
                    "x": pp.coordinates[location]["x"],
                    "y": pp.coordinates[location]["y"],
                    "value": max(0, data["cases"][datenum]),
                    "rolling_ave": max(0, data["rolling_ave"][datenum]),
                    "rate_of_change": max(0, data["rate_of_change"][datenum]),
                    "loc": location,
                    "prediction": is_pred,
                }
            )

    with open(save_location + "covidlocpreds.json", "w+") as f:
        return json.dump(new_dict, f)


def train(
    location,
    unique,
    counts,
    model_base_location,
    date_list=None,
    train_new_model=True,
    train_window=7,
    num_forecast=7,
):
    # pad with zeros for time series prediction
    if date_list is not None:
        inter_dates, inter_cases = pp.interpolate_cases(
            unique, counts, end=str(date_list[-1])
        )
    else:
        inter_dates, inter_cases = pp.interpolate_cases(unique, counts)

    # create a scaler for this location and normalize the data
    scaler = create_scaler()
    normalized = normalize_data(np.array(inter_cases), scaler)
    inout_seq = create_tensors(normalized, train_window)

    # train or load model
    if train_new_model:
        model = train_model(inout_seq)
        torch.save(model, model_base_location + location + ".pt")
    else:
        model = torch.load(model_base_location + location + ".pt")

    # make predictions
    normalized_preds = predict(model, num_forecast, normalized, train_window)
    predictions = denormalize_data(normalized_preds, scaler)

    # create date list for predictions
    prediction_dates = pp.get_date_list(
        start=str(datetime.strptime(
            inter_dates[-1], "%Y-%m-%d") + timedelta(days=1)),
        end=str(
            datetime.strptime(inter_dates[-1], "%Y-%m-%d")
            + timedelta(days=num_forecast)
        ),
    )

    # update case data with predictions
    inter_dates = np.append(inter_dates, prediction_dates)
    inter_cases = np.append(inter_cases, predictions)
    inter_dates = inter_dates.tolist()
    inter_cases = inter_cases.tolist()

    return location, inter_dates, inter_cases, prediction_dates, predictions


def _train(kwargs):
    return train(**kwargs)


def main(
    url,
    csv_location,
    model_base_location,
    download_new_file,
    locations,
    train_new_model,
    train_window,
    num_forecast,
    rolling_window,
    roc_window,
    roc_average,
    nproc,
):
    json_location = model_base_location

    # get data and create inout sequences
    if download_new_file:
        pp.download_csv(url, csv_location)

    if not locations:
        unique, counts = pp.process_csv(csv_location)
        _, dates, cases, pred_dates, pred_cases = train(
            location="model",
            unique=unique,
            counts=counts,
            model_base_location=model_base_location,
            train_new_model=train_new_model,
            train_window=train_window,
            num_forecast=num_forecast,
        )

        # plot original and predictions
        plt.title("Predictions")
        plt.ylabel("Confirmed Cases")
        plt.grid(True)
        plt.autoscale(axis="x", tight=True)
        plt.plot(dates, cases, label="actual")
        plt.plot(pred_dates, pred_cases, label="predictions")
        plt.plot(
            np.append(dates, pred_dates),
            pp.rolling_mean(np.append(cases, pred_cases), window=7),
            label="mean",
        )
        plt.legend()
        plt.show()
    else:
        # create dictionaries of data needed for each location
        locations_dict = pp.process_csv_locations(csv_location)
        date_list, _ = pp.process_csv(csv_location)
        output_dict = collections.defaultdict(dict)

        # train (or load) models for each location and get predictions
        _inputs = [
            dict(
                location=location,
                unique=unique,
                counts=counts,
                model_base_location=model_base_location,
                date_list=date_list,
                train_new_model=train_new_model,
                train_window=train_window,
                num_forecast=num_forecast,
            )
            for location, (unique, counts) in locations_dict.items()
        ]
        prediction_dates = []
        pool = torch.multiprocessing.Pool(nproc)
        for location, dates, cases, prediction_dates, _ in pool.imap_unordered(
            _train, _inputs, chunksize=1
        ):
            output_dict[location]["dates"] = dates
            output_dict[location]["cases"] = cases

        # calculate rolling mean and rate of change for each location
        for location, data in output_dict.items():
            output_dict[location]["rolling_ave"] = pp.rolling_mean(
                data["cases"], window=rolling_window
            )
            output_dict[location]["rate_of_change"] = pp.rate_of_change(
                data["cases"], window=roc_window, average=roc_average
            )

        # convert and save predictions to json
        convert_json(json_location, output_dict, prediction_dates)

        # plot original and predictions
        plt.close()
        for location in output_dict:
            plt.plot(
                output_dict[location]["dates"],
                output_dict[location]["cases"],
                label=location,
            )
        plt.title("COVID-19 Cases Per Location")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv",
    )
    parser.add_argument(
        "--csv_location",
        type=str,
        default=os.path.join(os.getcwd(), "data/conposcovidloc.csv"),
    )
    parser.add_argument(
        "--model_base_location", type=str, default=os.path.join(os.getcwd(), "data/")
    )
    parser.add_argument("--download_new_file", type=bool, default=False)
    parser.add_argument("--locations", type=bool, default=False)
    parser.add_argument("--train_new_model", type=bool, default=False)
    parser.add_argument("--train_window", type=int, default=7)
    parser.add_argument("--num_forecast", type=int, default=7)
    parser.add_argument("--rolling_window", type=int, default=7)
    parser.add_argument("--roc_window", type=int, default=7)
    parser.add_argument("--roc_average", type=bool, default=False)
    parser.add_argument("--nproc", type=int,
                        default=multiprocessing.cpu_count() - 2)
    args = parser.parse_args()
    main(**args.__dict__)
