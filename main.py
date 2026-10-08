from application import (
    BreastCancerDataProcessor,
    BreastCancerTrainer,
    BreastCancerPredictor,
    plot_history,
    print_usage,
    parse_args,
)

import sys

def process_phase(args: dict[str, str]):
    processor = BreastCancerDataProcessor()

    processor.load_data(args["dataset"])
    processor.split(test_size=0.2, random_state=42)
    processor.save(train_path="data_training.csv", valid_path="data_validation.csv")

    print("Dataset processed successfully.")
    print("Training dataset:   data_training.csv")
    print("Validation dataset: data_validation.csv")

def train_phase(args: dict[str, str]):
    trainer = BreastCancerTrainer(args)

    trainer.load_data(train_path=args["dataset"], valid_path=args["validation"])
    trainer.preprocess_data()

    history = trainer.train(epochs=int(args["epochs"]), batch_size=int(args["batch_size"]))
    model_path = args["model"]
    scaler_path = args["scaler"]

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
        print_usage()
        sys.exit(1)

