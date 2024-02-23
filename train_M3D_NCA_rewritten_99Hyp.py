
# %%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.models.Model_M3DNCA import M3DNCA
from src.models.Model_M3DNCA_alive import M3DNCA_alive
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
import time
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

config = [{
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Task99_HarP/imagesTr",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Task99_HarP/labelsTr",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Task97_DecathHip/imagesTr",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Task97_DecathHip/labelsTr",
    'name': r'M3D_NCA_Hyp97_v1', #12 or 13, 54 opt, 
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 3000,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [6, 16],
    'cell_fire_rate': 0.5,
    'batch_size': 20,
    'input_channels': 1,
    'output_channels': 2,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(9, 12, 10), (36, 48, 40)] ,
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': False,
    'rescale': True,
}
]

dataset = Dataset_NiiGz_3D(store=True)
device = torch.device(config[0]['device'])
ca1 = M3DNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels'], output_channels=config[0]['output_channels'], levels=2, scale_factor=4, steps=20).to(device)
ca = ca1
agent = M3DNCAAgent(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#agent.train(data_loader, loss_function)

### EVAL DATASETS
#agent.getAverageDiceScore(pseudo_ensemble=False)
print("--------------- TESTING HYP 99 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/home/jkalkhof_locale/Documents/Data/Task99_HarP/imagesTs", labelPath=r"/home/jkalkhof_locale/Documents/Data/Task99_HarP/labelsTs")
hyp99_test.exp = exp
#agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

print("--------------- TESTING HYP 98 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/home/jkalkhof_locale/Documents/Data/Task98_Dryad/imagesTs", labelPath=r"/home/jkalkhof_locale/Documents/Data/Task98_Dryad/labelsTs")
hyp99_test.exp = exp
#agent.getAverageDiceScore(pseudo_ensemble=True, dataset=hyp99_test)

print("--------------- TESTING HYP 97 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(36, 48, 40), imagePath=r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/imagesTr", labelPath=r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/labelsTr")
#hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/test/imagesTr/", labelPath=r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/test/labelsTr/")
#hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(64, 64, 48), imagePath=r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/test/imagesTr/", labelPath=r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/test/labelsTr/")

hyp99_test.exp = exp
#agent.getAverageDiceScore(pseudo_ensemble=True, dataset=hyp99_test)



start_time = time.perf_counter()
#agent.getAverageDiceScore(pseudo_ensemble=True)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")