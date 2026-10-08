VALID_PHASES = {"process", "train", "predict"}

VALID_PHASES = {
    "process",
    "train",
    "predict",
}

VALID_ACTIVATIONS = {
    "relu",
    "sigmoid",
    "tanh",
}

VALID_INITIALIZERS = {
    "random",
    "xavier",
    "he",
}

VALID_OPTIMIZERS = {
    "sgd",
    "adam",
}


DEFAULT_LAYERS = "24,24,24"
DEFAULT_ACTIVATION = "sigmoid"
DEFAULT_INITIALIZER = "xavier"
DEFAULT_OPTIMIZER = "sgd"

DEFAULT_EPOCHS = "84"
DEFAULT_BATCH_SIZE = "8"

DEFAULT_MODEL_PATH = "breast_cancer.model"
DEFAULT_SCALER_PATH = "breast_cancer.scaler"

DEFAULT_SGD_LEARNING_RATE = "0.01"
DEFAULT_ADAM_LEARNING_RATE = "0.001"

def print_usage():
    """
    Prints the usage information for the command-line interface.
    """
    print(
        "Usage:\n"
        "\n"
        "  Process raw dataset:\n"
        "    python main.py "
        "phase=process "
        "dataset=<raw_dataset>\n"
        "\n"
        "  Train model:\n"
        "    python main.py "
        "phase=train "
        "dataset=<training_dataset> "
        "validation=<validation_dataset> "
        "[model=<model_output_path>] "
        "[scaler=<scaler_output_path>] "
        "[layers=<size,size,...>] "
        "[activation=<relu|sigmoid|tanh>] "
        "[initializer=<random|xavier|he>] "
        "[optimizer=<sgd|adam>] "
        "[epochs=<number>] "
        "[batch_size=<number>] "
        "[learning_rate=<number>]\n"
        "\n"
        "  Training defaults:\n"
        f"    model={DEFAULT_MODEL_PATH}\n"
        f"    scaler={DEFAULT_SCALER_PATH}\n"
        f"    layers={DEFAULT_LAYERS}\n"
        f"    activation={DEFAULT_ACTIVATION}\n"
        f"    initializer={DEFAULT_INITIALIZER}\n"
        f"    optimizer={DEFAULT_OPTIMIZER}\n"
        f"    epochs={DEFAULT_EPOCHS}\n"
        f"    batch_size={DEFAULT_BATCH_SIZE}\n"
        f"    learning_rate={DEFAULT_SGD_LEARNING_RATE} for SGD, "
        f"{DEFAULT_ADAM_LEARNING_RATE} for Adam\n"
        "\n"
        "  Example:\n"
        "    python main.py "
        "phase=train "
        "dataset=data_training.csv "
        "validation=data_validation.csv "
        "layers=24,16,16 "
        "activation=sigmoid "
        "initializer=xavier "
        "optimizer=adam "
        "epochs=84 "
        "batch_size=8 "
        "learning_rate=0.001\n"
        "\n"
        "  Predict:\n"
        "    python main.py "
        "phase=predict "
        "dataset=<dataset> "
        "[model=<model_path>] "
        "[scaler=<scaler_path>]\n"
        "\n"
        "  Prediction defaults:\n"
        f"    model={DEFAULT_MODEL_PATH}\n"
        f"    scaler={DEFAULT_SCALER_PATH}\n"
    )


def parse_args(argv: list[str]) -> dict[str, str]:
    """
    Parses command-line arguments and validates them.
    
    args:
        list[str]: A list of command-line arguments in the form of key=value pairs.
    returns:
        dict[str, str]: A dictionary containing the parsed and validated arguments.
    raises:
        ValueError: If any argument is invalid or missing.
    """
    args: dict[str, str] = {}

    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Invalid argument: '{arg}'. Expected key=value.")

        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            raise ValueError(f"Invalid argument: '{arg}'.")

        if key in args:
            raise ValueError( f"Duplicate argument: '{key}'.")

        args[key] = value

    allowed = {
        "phase",
        "dataset",
        "validation",
        "model",
        "scaler",
        "layers",
        "activation",
        "initializer",
        "optimizer",
        "epochs",
        "batch_size",
        "learning_rate",
    }

    unknown = set(args) - allowed
    if unknown:
        raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")

    if "phase" not in args:
        raise ValueError("Missing required argument: phase")
    if args["phase"] not in VALID_PHASES:
        raise ValueError(
            f"Invalid phase '{args['phase']}'. "
            f"Expected one of: "
            f"{', '.join(sorted(VALID_PHASES))}"
        )
    if "dataset" not in args:
        raise ValueError("Missing required argument: dataset")

    phase = args["phase"]
    if phase == "process":
        forbidden = {
            "validation",
            "model",
            "scaler",
            "layers",
            "activation",
            "initializer",
            "optimizer",
            "epochs",
            "batch_size",
            "learning_rate",
        } & args.keys()

        if forbidden:
            raise ValueError(f"Process phase does not accept: {', '.join(sorted(forbidden))}")
    elif phase == "train":
        if "validation" not in args:
            raise ValueError("Training requires validation=<path_to_validation_dataset>")

        args.setdefault("initializer", DEFAULT_INITIALIZER)
        args.setdefault("activation", DEFAULT_ACTIVATION)
        args.setdefault("batch_size", DEFAULT_BATCH_SIZE)
        args.setdefault("optimizer", DEFAULT_OPTIMIZER)
        args.setdefault("scaler", DEFAULT_SCALER_PATH)
        args.setdefault("model", DEFAULT_MODEL_PATH)
        args.setdefault("layers", DEFAULT_LAYERS)
        args.setdefault("epochs", DEFAULT_EPOCHS)

        try:
            layers = [
                int(size.strip())
                for size
                in args["layers"].split(",")
            ]
        except ValueError:
            raise ValueError(
                "layers must contain "
                "comma-separated integers, "
                "e.g. layers=24,16,16"
            ) from None

        if len(layers) < 2:
            raise ValueError("At least two hidden layers are required.")
        if any(size <= 0 for size in layers):
            raise ValueError("All hidden layer sizes must be greater than 0.")

        activation = args["activation"].lower()
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Invalid activation "
                f"'{args['activation']}'. "
                f"Expected one of: "
                f"{', '.join(sorted(VALID_ACTIVATIONS))}"
            )
        args["activation"] = activation

        initializer = args["initializer"].lower()
        if initializer not in VALID_INITIALIZERS:
            raise ValueError(
                f"Invalid initializer "
                f"'{args['initializer']}'. "
                f"Expected one of: "
                f"{', '.join(sorted(VALID_INITIALIZERS))}"
            )
        args["initializer"] = initializer

        optimizer = args["optimizer"].lower()
        if optimizer not in VALID_OPTIMIZERS:
            raise ValueError(
                f"Invalid optimizer "
                f"'{args['optimizer']}'. "
                f"Expected one of: "
                f"{', '.join(sorted(VALID_OPTIMIZERS))}"
            )
        args["optimizer"] = optimizer

        if "learning_rate" not in args:
            if optimizer == "adam":
                args["learning_rate"] = DEFAULT_ADAM_LEARNING_RATE
            else:
                args["learning_rate"] = DEFAULT_SGD_LEARNING_RATE

        try:
            epochs = int(args["epochs"])
        except ValueError:
            raise ValueError("epochs must be an integer.") from None
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        try:
            batch_size = int(args["batch_size"])
        except ValueError:
            raise ValueError("batch_size must be an integer.") from None
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        try:
            learning_rate = float(args["learning_rate"])
        except ValueError:
            raise ValueError("learning_rate must be a number.") from None
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0.")

    elif phase == "predict":
        forbidden = {
            "validation",
            "layers",
            "activation",
            "initializer",
            "optimizer",
            "epochs",
            "batch_size",
            "learning_rate",
        } & args.keys()

        if forbidden:
            raise ValueError(f"Prediction phase does not accept: {', '.join(sorted(forbidden))}")

        args.setdefault("model", DEFAULT_MODEL_PATH)

        args.setdefault("scaler", DEFAULT_SCALER_PATH)

    return args
