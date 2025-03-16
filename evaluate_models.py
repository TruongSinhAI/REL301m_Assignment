from src.agents.dqn import DQNAgent
from src.env.env import BaseCarCrossingEnv
import torch

def load_model_dqn(filename):
    agent_ai = DQNAgent(19, 5)
    agent_ai.epsilon = 0
    agent_ai.policy_net.load_state_dict(torch.load(filename, weights_only=True))
    agent_ai.policy_net.eval()
    return agent_ai


env = BaseCarCrossingEnv()
agent_ai = load_model_dqn("dqn_agent_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
win_count = 0
for turn in range(100):
    done = False
    observation = env.reset()
    while not done:
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent_ai.policy_net(state_tensor)
        action = q_values.argmax().item()
        observation, reward, done, info = env.step(action)
    
    if env.score>=100:
        win_count+=1
    print(turn)
print(win_count) #75