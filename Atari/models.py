
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    # def try_load(self, save_dir):
    #     paths = glob.glob(save_dir + '*.tar') ; step = 0
    #     if len(paths) > 0:
    #         ckpts = [int(s.split('.')[-2]) for s in paths]
    #         ix = np.argmax(ckpts) ; step = ckpts[ix]
    #         self.load_state_dict(torch.load(paths[ix]))
    #     print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
    #     return step

# model=NNPolicy(1,256,4)
# for name, param in model.named_parameters():
#     print (name)


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.fc1 = nn.Linear(21587, 2000)
        self.fc2 = nn.Linear(2000,100)
        self.fc3 = nn.Linear(100,2000)
        self.fc4 = nn.Linear(2000, 21587)
        self.sigmoid = nn.Tanh()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 21587)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.fc4(h2)

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2


class CAE_Big(nn.Module):
    def __init__(self):
        super(CAE_Big, self).__init__()
        self.fc1 = nn.Linear(76863, 3000)
        self.fc2 = nn.Linear(3000,200)
        self.fc3 = nn.Linear(200,3000)
        self.fc4 = nn.Linear(3000, 76863)
        self.sigmoid = nn.Tanh()
        #self.Linear = nn.Linear()

    def encoder(self, x):
        h1 = self.sigmoid(self.fc1(x.view(-1, 76863)))
        return self.sigmoid(self.fc2(h1))

    def decoder(self, z):
        h2 = self.sigmoid(self.fc3(z))
        return self.fc4(h2)

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2