import argparse
import json

from dataset import get_dataset
import model as Model
import trainer as Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EmotionAnalyzer, a final project for Intro to AI')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path.')
    options = parser.parse_args()
    config_path = options.config
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_iter, test_iter = get_dataset(config)
    model = getattr(Model, config['model'])(config)
    trainer = Trainer.trainer(config, train_iter, test_iter, model)

    trainer.train()