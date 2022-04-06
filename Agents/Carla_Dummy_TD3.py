import gym
import gym_carla

import random
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("CarlaIL-v0") 

for ep in range(30):

    obs = env.reset()
    print(obs.shape)
    p = obs

    for i in range(300):
        next_obs, rew, done, info = env.step([2*random.random()-1,random.random()])
        obs = next_obs
        if done:
            break

env.close()
print("Goodbye, sir!")

# p = p[:6*100*200].reshape(6,100,200)[:3,:,:]/255.0
# print(p.shape, type(p), p.dtype)
# # print(p.max())
# plt.imshow(p.reshape(100,200,3))
# plt.show()






# # network architecture
# class Network(nn.Module):
    
#     def __init__(self):
#         super(Network, self).__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
#             nn.BatchNorm2d(num_features=32),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=32),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
#             nn.BatchNorm2d(num_features=64),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=64),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
#             nn.BatchNorm2d(num_features=128),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=128),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=256),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(num_features=256),
#             nn.Dropout2d(),
#             nn.ReLU(),)
#         self.flatten = nn.Flatten()
#         self.dense1 = nn.Sequential(
#             nn.Linear(in_features=8192, out_features=512),
#             nn.ReLU())
#         self.dense2 = nn.Sequential(
#             nn.Linear(in_features=512, out_features=512),
#             nn.ReLU())
#         self.output = nn.Linear(512, 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dense2(x) 
#         x = self.output(x)
#         return x

# model = Network()

# model.load_state_dict(torch.load('/home/gav/Dev/Carla/model/model_weights_20220312_1821.pth'))
# model.eval()
# # # a = model(torch.zeros((1,3,88,200)))
# # # print(float(a[0,0]), a[0,1])
