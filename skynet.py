import arguments_parser
from result_predictor import ResultPredictor
from network_trainer import NetworkTrainer
from data_transformer import DataTransformer
from file_handler import FileHandler


def main():
    data_transformer = DataTransformer()
    file_handler = FileHandler()
    result_predictor = ResultPredictor(data_transformer=data_transformer, file_handler= file_handler)
    network_trainer = NetworkTrainer(data_transformer=data_transformer, file_handler=file_handler)

    args = arguments_parser.parse()

    match args.action:
        case 'train':
            network_trainer.train()
        case 'predict':
            result_predictor.predict(args.other_args)


if __name__ == '__main__':
    main()
