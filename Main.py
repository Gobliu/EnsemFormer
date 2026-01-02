import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--learning_rate', type=float) # No default here!
    parser.add_argument('--epochs', type=int)
    return parser.parse_args()

def main():
    cli_args = get_args()
    
    # 1. Load YAML
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Override YAML with CLI if user provided them
    # vars(cli_args) converts the namespace to a dictionary
    for key, value in vars(cli_args).items():
        if value is not None:
            # You can decide how deep your nesting goes here
            if key in config['training']:
                config['training'][key] = value

    # 3. Pass the 'config' dict to your PermeaFold 4D module