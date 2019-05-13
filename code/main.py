import argparse
import json

import dataset as Dataset
import model as Model
import trainer as Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EmotionAnalyzer, a final project for Intro to AI')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path.')
    config_path = parser.parse_args()['config']
    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset = getattr(Dataset, config['representation'])(config)
    model = getattr(Model, config['model'])(config)
    trainer = trainer.trainer(config, dataset, model)

    trainer.train()