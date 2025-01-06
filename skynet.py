import argparse
from result_predictor import ResultPredictor
from network_trainer import NetworkTrainer
from data_transformer import DataTransformer
from file_handler import FileHandler

EPOCHS_NUMBER = 3

def main():
    data_transformer = DataTransformer()
    file_handler = FileHandler()
    result_predictor = ResultPredictor(data_transformer=data_transformer, file_handler= file_handler)
    network_trainer = NetworkTrainer(data_transformer=data_transformer, file_handler=file_handler)

    parser = argparse.ArgumentParser(description='Pass action with arguments')
    parser.add_argument("action", type=str, help="action")
    parser.add_argument('other_args', nargs='*', help='Other arguments')
    args = parser.parse_args()

    match args.action:
        case 'train':
            network_trainer.train(EPOCHS_NUMBER)
        case 'predict':
            result_predictor.predict(args.other_args)


if __name__ == '__main__':
    main()
