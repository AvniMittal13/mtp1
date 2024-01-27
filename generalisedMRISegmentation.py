#%%
import torch
from src.datasets.Nii_Gz_Dataset_3D_gen import Dataset_NiiGz_3D_gen
from src.models.Model_GenNCA import GenNCA
from src.models.Model_GenNCA_v2 import GenNCA_v2
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_NCA_gen import Agent_NCA_gen

from src.models.Model_BasicNCA3D import BasicNCA3D
from src.agents.Agent_NCA import Agent_NCA
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


config = [{
    # Basic
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus_Prostate_Liver/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus_Prostate_Liver/labelsTr/",
    'name': r"genMRIseg_114_v2_noShift_hipProsLiver",#_baseline", 75% with vec, 77.5% baseline
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 10,#
    'evaluate_interval': 10,
    'n_epoch': 1000,
    'batch_size': 8,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': 20,
    'cell_fire_rate': 0.5,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'extra_channels': 4,
    # Data
    'input_size': (42, 42, 12),
    'data_split': [0.7, 0, 0.3], 
}
]
dataset = Dataset_NiiGz_3D_gen(extra_channels=config[0]['extra_channels'])
device = torch.device(config[0]['device'])

ca = GenNCA_v2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], extra_channels=config[0]['extra_channels'], kernel_size=3).to(device)
agent = Agent_NCA_gen(ca)

#ca = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels']).to(device)
#agent = Agent_NCA_gen(ca)

exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceFocalLoss() 

#agent.train(data_loader, loss_function)

agent.getAverageDiceScore()



# %%