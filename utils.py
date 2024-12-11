import configparser
import os
import sys
from sumolib import checkBinary


def load_train_config(config_file):
    parser = configparser.ConfigParser()
    parser.read(config_file)
    config = {
        'gui': parser['simulation'].getboolean('gui'),
        'total_episodes': parser['simulation'].getint('total_episodes'),
        'max_steps': parser['simulation'].getint('max_steps'),
        'n_cars_generated': parser['simulation'].getint('n_cars_generated'),
        'green_duration': parser['simulation'].getint('green_duration'),
        'yellow_duration': parser['simulation'].getint('yellow_duration'),
        'num_layers': parser['model'].getint('num_layers'),
        'width_layers': parser['model'].getint('width_layers'),
        'batch_size': parser['model'].getint('batch_size'),
        'learning_rate': parser['model'].getfloat('learning_rate'),
        'training_epochs': parser['model'].getint('training_epochs'),
        'memory_size_min': parser['memory'].getint('memory_size_min'),
        'memory_size_max': parser['memory'].getint('memory_size_max'),
        'num_states': parser['agent'].getint('num_states'),
        'num_actions': parser['agent'].getint('num_actions'),
        'gamma': parser['agent'].getfloat('gamma'),
        'models_path_name': parser['dir']['models_path_name'],
        'sumocfg_file_name': parser['dir']['sumocfg_file_name']
    }
    return config


def load_test_config(config_file):
    parser = configparser.ConfigParser()
    parser.read(config_file)
    config = {
        'gui': parser['simulation'].getboolean('gui'),
        'max_steps': parser['simulation'].getint('max_steps'),
        'n_cars_generated': parser['simulation'].getint('n_cars_generated'),
        'episode_seed': parser['simulation'].getint('episode_seed'),
        'green_duration': parser['simulation'].getint('green_duration'),
        'yellow_duration': parser['simulation'].getint('yellow_duration'),
        'num_states': parser['agent'].getint('num_states'),
        'num_actions': parser['agent'].getint('num_actions'),
        'sumocfg_file_name': parser['dir']['sumocfg_file_name'],
        'models_path_name': parser['dir']['models_path_name'],
        'model_to_test': parser['dir'].getint('model_to_test')
    }
    return config


def configure_sumo(gui, sumocfg_file_name, max_steps):
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Environment variable 'SUMO_HOME' is not defined. Please set it before proceeding.")

    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

    # Determine whether to use SUMO or SUMO-GUI
    sumo_binary = checkBinary('sumo-gui') if gui else checkBinary('sumo')

    # Create the SUMO command
    sumo_cmd = [
        sumo_binary,
        "-c", os.path.join('environment', sumocfg_file_name),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps)
    ]
    return sumo_cmd


def create_train_path(models_path_name):
    models_dir = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(models_dir, exist_ok=True)

    existing_versions = [
        int(name.split("_")[1])
        for name in os.listdir(models_dir) if name.startswith("model_")
    ]

    new_version = str(max(existing_versions) + 1) if existing_versions else '1'

    new_path = os.path.join(models_dir, f'model_{new_version}', '')
    os.makedirs(new_path, exist_ok=True)
    return new_path


def create_test_path(models_path_name, model_number):
    model_folder_path = os.path.join(os.getcwd(), models_path_name, f'model_{model_number}', '')

    if not os.path.isdir(model_folder_path):
        sys.exit(f"Model folder 'model_{model_number}' does not exist in '{models_path_name}'.")

    test_folder_path = os.path.join(model_folder_path, 'test', '')
    os.makedirs(test_folder_path, exist_ok=True)
    return model_folder_path, test_folder_path
