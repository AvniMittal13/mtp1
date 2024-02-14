
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.models.Model_ViTCA3D import BasicViTNCA3D
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3D_NCA import Agent_M3D_NCA
config = [{
    'img_path': r"/content/3d-liver-segmentation/Task03_Liver_rs/imagesTr",
    'label_path': r"/content/3d-liver-segmentation/Task03_Liver_rs/labelsTr",
    'name': r'M3D_NCA_Run5',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 3000,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [10, 10],
    'cell_fire_rate': 0.5,
    'batch_size': 4,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(16, 16, 13),(64, 64, 52)], # 
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': True,
    'rescale': True,
}
]
dataset = Dataset_NiiGz_3D()
device = torch.device(config[0]['device'])
ca1 = BasicViTNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels']).to(device)
ca2 = BasicViTNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
ca = [ca1, ca2]
agent = Agent_M3D_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() 

agent.train(data_loader, loss_function)

agent.getAverageDiceScore()

