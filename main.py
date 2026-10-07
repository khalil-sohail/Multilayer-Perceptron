from application import (
    BreastCancerDataProcessor,
    BreastCancerTrainer,
    BreastCancerPredictor,
    plot_history
)

import sys

VALID_PHASES = {"process", "train", "predict"}

def print_usage():
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
        "[scaler=<scaler_output_path>]\n"
        "\n"
        "  Predict / evaluate:\n"
        "    python main.py "
        "phase=predict "
        "dataset=<dataset> "
        "model=<model_path> "
        "scaler=<scaler_path>\n"
    )


def parse_args(argv: list[str]) -> dict[str, str]:
    args: dict[str, str] = {}

    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Invalid argument: '{arg}'")

        key, value = arg.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            raise ValueError(f"Invalid argument: '{arg}'")
        if key in args:
            raise ValueError(f"Duplicate argument: '{key}'")

        args[key] = value

    allowed = {
        "phase",
        "dataset",
        "validation",
        "model",
        "scaler",
    }

    unknown = set(args) - allowed
    if unknown:
        raise ValueError(f"Unknown argument(s): {', '.join(sorted(unknown))}")

    if "phase" not in args:
        raise ValueError("Missing required argument: phase")
    if args["phase"] not in VALID_PHASES:
        raise ValueError(
            f"Invalid phase '{args['phase']}'. "
            f"Expected one of: {', '.join(sorted(VALID_PHASES))}"
        )
    if "dataset" not in args:
        raise ValueError("Missing required argument: dataset")

    phase = args["phase"]
    if phase == "process":
        forbidden = {"validation", "model", "scaler"} & args.keys()
        if forbidden:
            raise ValueError(f"Process phase does not accept: {', '.join(sorted(forbidden))}")

    elif phase == "train":
        if "validation" not in args:
            raise ValueError("Training requires validation=<path_to_validation_dataset>")

    elif phase == "predict":
        if "model" not in args:
            raise ValueError("Prediction requires model=<path_to_model_file>")

        if "scaler" not in args:
            raise ValueError("Prediction requires scaler=<path_to_scaler_file>")

        if "validation" in args:
            raise ValueError("Prediction does not accept a validation dataset.")

    return args


def process_phase(args: dict[str, str]):
    processor = BreastCancerDataProcessor()

    processor.load_data(args["dataset"])
    processor.split(test_size=0.2, random_state=42)
    processor.save(train_path="data_training.csv", valid_path="data_validation.csv")

    print("Dataset processed successfully.")
    print("Training dataset:   data_training.csv")
    print("Validation dataset: data_validation.csv")


def train_phase(args: dict[str, str]):
    trainer = BreastCancerTrainer()

    trainer.load_data(train_path=args["dataset"], valid_path=args["validation"])
    trainer.preprocess_data()
    history = trainer.train(epochs=100, batch_size=1)
    model_path = args.get("model", "breast_cancer.model")
    scaler_path = args.get("scaler", "breast_cancer.scaler")

    trainer.save(model_path=model_path, scaler_path=scaler_path)

    print(f"Model saved to:  {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    plot_history(history)


def predict_phase(args: dict[str, str]):
    predictor = BreastCancerPredictor(model_path=args["model"], scaler_path=args["scaler"])

    predictor.load_data(args["dataset"])
    predictor.preprocess_data()
    results = predictor.evaluate()

    print("Evaluation:")
    print(f"  Loss:      {results['loss']:.6f}")
    print(f"  Accuracy:  {results['accuracy']:.6f}")
    print(f"  Precision: {results['precision']:.6f}")
    print(f"  Recall:    {results['recall']:.6f}")
    print(f"  F1 score:  {results['f1_score']:.6f}")


def Main(args: dict[str, str]):
    phase = args["phase"]

    if phase == "process":
        process_phase(args)

    elif phase == "train":
        train_phase(args)

    elif phase == "predict":
        predict_phase(args)


if __name__ == "__main__":
    try:
        args = parse_args(sys.argv[1:])
        Main(args)

    except Exception as error:
        print(f"Error: {error}\n")
        import traceback
        traceback.print_exc()
        print_usage()
        sys.exit(1)

