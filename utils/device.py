import torch


def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "Using GPU:",
            torch.cuda.get_device_name(0),
        )

    else:

        device = torch.device("cpu")

        print("Using CPU")

    return device