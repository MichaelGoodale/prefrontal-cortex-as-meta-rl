from models import PrefrontalLSTM
from training import train
import matplotlib.pyplot as plt
from tqdm import tqdm

model = PrefrontalLSTM(4, 2)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
env = gym.make('CartPole-v1')
loss = []
rewards = []
for i in tqdm(range(2000)):
    l, r = train(env, model, optimizer)
    loss.append(l)
    rewards.append(r)

env.close()

plt.plot(loss)
plt.plot(r)

plt.show()
