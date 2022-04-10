import torch
import torch.nn as nn

'''
class MultiModActor(nn.Module):
    """Multi-Mod actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.
    """

    def __init__(self):
        super(MultiModActor, self).__init__()
        self.device = 'cuda'

        in_channels = [6,32,64,64]
        vision_model = []
        vision_model += [nn.Conv2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=5, stride=(2,4), padding=1)]
        vision_model += [nn.ReLU()]
        vision_model += [nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, stride=2, padding=0)]
        vision_model += [nn.ReLU()]
        vision_model += [nn.Conv2d(in_channels=in_channels[2], out_channels=64            , kernel_size=2, stride=1, padding=1)]
        vision_model += [nn.ReLU()]

        in_channels = [64*22*25+3+6,512,256]
        critic_model = []
        critic_model += [nn.Linear(in_features=in_channels[0], out_features=in_channels[1])]
        critic_model += [nn.ReLU()]
        critic_model += [nn.Linear(in_features=in_channels[1], out_features=2)]
        critic_model += [nn.ReLU()]

        self.vision_process = nn.Sequential(*vision_model)
        self.last = nn.Sequential(*critic_model)

    def forward(self, obs, state, info=None):    
        """Mapping: obs -> logits -> action."""
        # extract the sub-state
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        val = obs[:, 6*88*200:]
        # preproccess the image feature
        x = obs[:,:6*88*200].reshape((-1,6,88,200))
        x = self.vision_process(x).flatten(1)
        # concatenate the state and action
        x = torch.concat((x, val), dim=1)
        # preprocess all the state and action 
        x = self.last(x)

        return x, state # return logits, hidden

class MultiModCritic(nn.Module):
    """Multi-Mod critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).
    """
    
    def __init__(self):
        super(MultiModCritic, self).__init__()
        self.device = 'cuda'

        in_channels = [6,32,64,64]
        vision_model = []
        vision_model += [nn.Conv2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=5, stride=(2,4), padding=1)]
        vision_model += [nn.ReLU()]
        vision_model += [nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, stride=2, padding=0)]
        vision_model += [nn.ReLU()]
        vision_model += [nn.Conv2d(in_channels=in_channels[2], out_channels=64            , kernel_size=2, stride=1, padding=1)]
        vision_model += [nn.ReLU()]

        in_channels = [64*22*25+3+6+2,512]
        critic_model = []
        critic_model += [nn.Linear(in_features=in_channels[0], out_features=in_channels[1])]
        critic_model += [nn.ReLU()]
        critic_model += [nn.Linear(in_features=in_channels[1], out_features=2)]
        critic_model += [nn.ReLU()]

        self.vision_process = nn.Sequential(*vision_model)
        self.last = nn.Sequential(*critic_model)

    def forward(self, obs, act, info=None) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        # extract the sub-state
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        val = obs[:, 6*88*200:]
        # reshape the action
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
        # preproccess the image feature
        x = obs[:,:6*88*200].reshape((-1,6,88,200))
        x = self.vision_process(x).flatten(1)
        # concatenate the state and action
        x = torch.concat((x, val, act), dim=1)
        # preprocess all the state and action 
        x = self.last(x)
        
        return x # return logits
'''


class MultiModActor(nn.Module):
    """Multi-Mod actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.
    """

    def __init__(self):
        super(MultiModActor, self).__init__()
        self.device = 'cuda'
        self.scaling = 1.0,

        channels = [6,32,32,64,64,128,128] # ,256,256]
        kernel_sizes = [5, 5, 3, 3, 3, 3] #, 3, 3]
        stride_sizes = [(2,2), (1,2), 1, 1, 1, 1] #,1, 1]
        # padding_sizes = [0,0,0,0,0,0,0,0]
        vision_model = []
        for i in range(len(channels)-1):
            vision_model += [
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_sizes[i], stride=stride_sizes[i], padding=0,),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.Dropout2d(),
                nn.ReLU(),
            ]
        
        channels = [128*36*39,512,128,32,2]
        mlp_model = []
        for i in range(2):
            mlp_model += [
                nn.Linear(in_features=channels[i], out_features=channels[i+1]),
                nn.ReLU()]
        
        out_model = []
        out_model += [nn.Linear(in_features=channels[2]+9, out_features=channels[3])]
        out_model += [nn.Dropout()]
        out_model += [nn.ReLU()]
        out_model += [nn.Linear(in_features=channels[3], out_features=channels[4])]

        self.conv = nn.Sequential(*vision_model)
        self.mlp = nn.Sequential(*mlp_model)
        self.out = nn.Sequential(*out_model)

    def forward(self, obs, state, info=None):    
        """Mapping: obs -> logits -> action."""
        # extract the sub-state
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        val = obs[:, 6*100*200:]
        # # preproccess the image feature
        x = obs[:,:6*100*200].reshape((-1,6,100,200))
        x = self.conv(x)
        x = x.flatten(1) # [1, 256, 34, 36] -> [1, 313344]
        x = self.mlp(x)
        # # concatenate the state and action
        x = torch.concat((x, val), dim=1)
        # # preprocess all the state and action 
        x = self.out(x)
        x = 2*torch.sigmoid(x) - 1
       
        return x, state # return logits, hidden

class MultiModActor2(nn.Module):
    """Multi-Mod actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.
    """

    def __init__(self):
        super(MultiModActor2, self).__init__()
        self.device = 'cuda'
        self.scaling = 1.0,

        channels = [6,32,32,64,64,128,128] # ,256,256]
        kernel_sizes = [5, 5, (3,3), (3,3), 2, 2] #, 3, 3]
        stride_sizes = [(3,3), (1,2), 1, 1, 1, 1] #,1, 1]

        vision_model = []
        for i in range(len(channels)-1):
            vision_model += [
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_sizes[i], stride=stride_sizes[i], padding=0,),
                nn.BatchNorm2d(num_features=channels[i+1]),
                nn.Dropout2d(),
                nn.ReLU(),
            ]
        
        channels = [128*22*25,512,128,27]
        mlp_model = []
        for i in range(len(channels)-1):
            mlp_model += [
                nn.Linear(in_features=channels[i], out_features=channels[i+1]),
                nn.Dropout(),
                nn.ReLU()]
        
        out_model = []
        out_model += [nn.Linear(in_features=channels[-1]+9, out_features=2)]


        self.conv = nn.Sequential(*vision_model)
        self.mlp = nn.Sequential(*mlp_model)
        self.out = nn.Sequential(*out_model)

    def forward(self, obs, state, info=None):    
        """Mapping: obs -> logits -> action."""
        # extract the sub-state
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        val = obs[:, 6*100*200:]
        # # preproccess the image feature
        x = obs[:,:6*100*200].reshape((-1,6,100,200))
        x = self.conv(x)
        x = x.flatten(1) # [1, 128, 22, 25] -> flatten()
        x = self.mlp(x)
        # # concatenate the state and action
        x = torch.concat((x, val), dim=1)
        # # preprocess all the state and action 
        x = self.out(x)
        x = 2*torch.sigmoid(x) - 1
       
        return x, state # return logits, hidden

class MultiModCritic(nn.Module):
    """Multi-Mod critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).
    """
    
    def __init__(self):
        super(MultiModCritic, self).__init__()
        self.device = 'cuda'

        in_channels = [6,32,64,64]
        vision_model = []
        vision_model += [nn.Conv2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=5, stride=(2,4), padding=1)]
        vision_model += [nn.ReLU()]
        vision_model += [nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, stride=2, padding=0)]
        vision_model += [nn.ReLU()]
        vision_model += [nn.Conv2d(in_channels=in_channels[2], out_channels=64            , kernel_size=2, stride=1, padding=1)]
        vision_model += [nn.ReLU()]

        in_channels = [64*25*25,512,256,128,64,32,2]
        mlp_model = []
        mlp_model += [nn.Linear(in_features=in_channels[0], out_features=in_channels[1])]
        mlp_model += [nn.ReLU()]
        mlp_model += [nn.Linear(in_features=in_channels[1], out_features=in_channels[2])]
        mlp_model += [nn.ReLU()]
        mlp_model += [nn.Linear(in_features=in_channels[2], out_features=in_channels[3])]
        mlp_model += [nn.ReLU()]
        mlp_model += [nn.Linear(in_features=in_channels[3], out_features=in_channels[4])]
        mlp_model += [nn.ReLU()]

        out_model = []
        out_model += [nn.Linear(in_features=in_channels[4]+9+2, out_features=in_channels[5])]
        out_model += [nn.ReLU()]
        out_model += [nn.Linear(in_features=in_channels[5], out_features=in_channels[6])]

        self.conv = nn.Sequential(*vision_model)
        self.mlp = nn.Sequential(*mlp_model)
        self.out = nn.Sequential(*out_model)

    def forward(self, obs, act, info=None) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        # extract the sub-state
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        val = obs[:, 6*100*200:]
        # reshape the action
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
        # preproccess the image feature
        x = obs[:,:6*100*200].reshape((-1,6,100,200))
        x = self.conv(x).flatten(1) # [1, 256, 34, 36] -> [1, 120000]
        x = self.mlp(x)
        # concatenate the state and action
        x = torch.concat((x, val, act), dim=1)
        # preprocess all the state and action 
        x = self.out(x)
        
        return x # return logits


if __name__ == "__main__":
    inp = torch.randn(10,6*100*200+9)
#     print(inp.shape)

    net = MultiModActor2().to("cuda")
    print(net)
    out, _ = net(inp, None)
#     # crt = MultiModCritic().to("cuda")
#     # out = crt(inp, torch.ones(1,2))
    # print(out.shape)
    print(out)