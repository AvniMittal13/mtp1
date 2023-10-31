import os
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3D_NCA import Agent_M3D_NCA

def segment_nii_files(input_directory, class_name, output_directory, device, model_id):
    # Set the output directory to default if not provided
    if output_directory is None:
        output_directory = os.path.join(input_directory, "segmentation_results")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    dataset = Dataset_NiiGz_3D()
    device = torch.device(device)

    # get model config from model_id and class
    # ca = [list of cas]
    # config  -> all configs have datasplit [0,0,1]
    config, ca = get_model_details(class_name, model_id, device)

    agent = Agent_M3D_NCA(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('test')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    agent.getAverageDiceScore()

    # load model weights

    # make predictions and save



    # # Process .nii files in the input directory
    # for file_name in os.listdir(input_directory):
    #     if file_name.endswith(".nii"):
    #         file_path = os.path.join(input_directory, file_name)
    #         # Implement your segmentation logic here
    #         # You can use ITK, deep learning models, or any other method
    #         # Save the segmentation results in the output_directory


def get_model_details(class_name, model_id, device):
    path = os.path.join('models', class_name, model_id,'config.dt')
    config = load_json_file(path)
    config[0]['data_split'] = [0, 0, 1]
    num_ncas = config[0]['train_model']+1
    ca = [BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=i*4 + 3, input_channels=config[0]['input_channels']).to(device)
          for i in range(num_ncas)]
    return config, ca


def perform_segmentation(input_image, class_name, device):


def save_nii_image(image, output_path):
    # Save the NIfTI image to the specified path
    nib.save(image, output_path)
