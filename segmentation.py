import os
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3D_NCA import Agent_M3D_NCA
from default_models import model_mappings

def segment_nii_files(input_directory, class_name, output_directory, device, model_id, pseudo_ensemble, variance_map):
    # Set the output directory to default if not provided
    if output_directory is None:
        output_directory = os.path.join(input_directory, "segmentation_results")
    if pseudo_ensemble and variance_map:
        variance_maps_directory = os.path.join(output_directory, "variance_maps")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok = "True")
    if pseudo_ensemble and variance_map:
        os.makedirs(variance_maps_directory, exist_ok = "True")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    dataset = Dataset_NiiGz_3D()
    device = torch.device(device)

    # get model config from model_id and class
    # ca = [list of cas]
    # config  -> all configs have datasplit [0,0,1]
    config, ca = get_model_details(class_name, model_id, device)

    config[0]['img_path'] = input_directory
    config[0]['label_path'] = input_directory
    config[0]['input_size'] = [tuple(sublist) for sublist in config[0]['input_size']]
    config[0]['output_path'] = output_directory
    if pseudo_ensemble and variance_map:
        config[0]['variance_maps_path'] = variance_maps_directory

    print("img_path: ", config[0]['img_path'])
    print("label_path: ", config[0]['label_path'])     
    print("len ca: ", len(ca))
    agent = Agent_M3D_NCA(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    agent.predict(pseudo_ensemble = pseudo_ensemble)
    # agent.getAverageDiceScore()


def get_model_details(class_name, model_id, device):
    model_id = update_model_id(class_name, model_id)
    path = os.path.join('models', class_name, model_id,'config.dt')
    config = load_json_file(path)
    config[0]['data_split'] = [0, 0, 1]
    num_ncas = config[0]['train_model']+1
    ca = [BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=(num_ncas-i)*4 -1, input_channels=config[0]['input_channels']).to(device)
          for i in range(num_ncas)]
    ca.reverse()
    return config, ca

def update_model_id(class_name, model_id=None):
    if model_id is None:
        model_id = model_mappings.get(class_name)

    return model_id

# def perform_segmentation(input_image, class_name, device):


# def save_nii_image(image, output_path):
#     # Save the NIfTI image to the specified path
#     nib.save(image, output_path)
