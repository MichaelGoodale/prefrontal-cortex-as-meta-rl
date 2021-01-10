#File where we define any necessary tasks for agent to learn
import random
import gym
import utils


class TaskOne:
    def __init__(self, reward=1, seed=1337, mode='bandit'):

        if mode not in ['bandit', 'monkey', 'bandit_uncorr']:
            raise Exception(f"Invalid mode, {mode} must be either 'bandit', 'bandit_uncorr' or 'monkey'")

        self.mode = mode
        self.random_generator = random.Random(seed)
        self.reward = reward

        self.train = True

        #left, right
        self.initial_probability = [0,0]
        self.probability_reward = [0,0]
        self.time_since_action = [0,0]

        self.reset()

    def set_train(self):
        self.train = True

    def set_test(self):
        self.train = False

    def reset(self, initial_probs=None, max_trials=100):
        self.trial = 0

        if self.mode == 'monkey':
            if self.train:
                self.initial_probability[0] = self.random_generator.choice([self.random_generator.uniform(0,0.1), self.random_generator.uniform(0.2, 0.3), self.random_generator.uniform(0.4,0.5)])
                self.trials = self.random_generator.randint(50,100)
            else:
                self.initial_probability[0] = self.random_generator.uniform(0,0.5) #Is it the complement of train or uniform? P.15
                self.trials = 100

            self.initial_probability[1] = 0.5 - self.initial_probability[0]

            if self.train and self.random_generator.random() <= 0.5: #Avoid biasing one arm.
                self.initial_probability = self.initial_probability[::-1]
        else:
            self.initial_probability[0] = self.random_generator.random()
            if self.mode == 'bandit_uncorr':
                self.initial_probability[1] = self.random_generator.random()
            else:
                self.initial_probability[1] = 1.0 - self.initial_probability[0]
            self.trials = max_trials


        if initial_probs is not None:
            self.initial_probability = initial_probs

        self.probability_reward = self.initial_probability.copy()
        self.time_since_action = [0,0]
        return []

    def step(self, action):
        if self.random_generator.uniform(0, 1) <= self.probability_reward[action]:
            r = self.reward
        else:
            r = 0

        if self.mode == 'monkey':
            self.time_since_action[action] = 0
            self.time_since_action[(action+1)%2] += 1

            self.probability_reward = [1-(1-self.initial_probability[0])**(self.time_since_action[0]+1), \
                                       1-(1-self.initial_probability[1])**(self.time_since_action[1]+1)]

        self.trial += 1

        if self.trial >= self.trials:
            done = True
        else:
            done = False
        return [], r, done, None


class TwoStep:
    def __init__(self, reward=1, seed=1337, probability_transition=0.8, probability_reward=0.9, switch_rate=0.025, trials=100):

        self.probability_transition = probability_transition
        self.switch_rate = switch_rate
        self.reward = reward
        self.trials = trials

        #dim 0 is state, dim 1 probability of reward for an action given a state,  may want to add asymmetricality
        self.probability_reward = [[probability_reward, 1-probability_reward], [1-probability_reward, probability_reward]]

        self.random_generator = random.Random(seed)

        self.reset()

    def shuffle(self, p):
        #shuffles probability of reward
        if self.random_generator.uniform(0,1)<=p:
            self.probability_reward.reverse()

    def reset(self):
        self.trial = 0
        self.state = [0,0]
        self.stage = 0
        self.shuffle(0.5)

    def step(self, action):
        if self.stage == 0:

            self.shuffle(self.switch_rate)

            if self.random_generator.uniform(0,1) <= self.probability_transition:
                self.state[action] = 1
            else:
                self.state[(action+1)%2] = 1

            r = 0
            self.stage = 1

        else:
            if self.random_generator.uniform(0,1) <= self.probability_reward[self.state.index(1)][action]:
                r = self.reward
            else:
                r = 0

            self.state = [0,0]
            self.stage = 0

        self.trial += 1

        if self.trial >= self.trials:
            done = True
        else:
            done = False

        return self.state, r, done, None

class TwoStepsGridWorld:
    def __init__(self, seed=1337, probability_transition=0.8, probability_reward=0.9, switch_rate=0.025, trials=100):

        self.random_generator = random.Random(seed)

        self.probability_transition = probability_transition
        self.probability_reward = probability_reward
        self.switch_rate = switch_rate
        self.trials = trials

        self.trial = 0
        self.stage = 0

        #from GridWorld.gridworld_env import GridworldEnv
        from gym.envs.registration import register
        register(
            id='gridworld-v0',
            entry_point='GridWorld.gridworld_env:GridworldEnv',
        )

        self.map0 = "gridworldPlans/plan-two-steps-pre.txt"
        self.rewards = {0: -0.001, 3: 0, 4: 1, 5: 0, 6: 0}

        self.pos1 = [4,1]
        self.pos2 = [4,5]
        self.pos_post = [[1,1], [1,5]]
        self.rew = [{0: -0.001, 3: 0, 4: 0, 5: 1, 6: 0}, {0: -0.001, 3: 0, 4: 0, 5: 0, 6: 1}]

        self.env = gym.make("gridworld-v0")
        self.env.setPlan(self.map0, self.rewards)
        self.env.seed(seed)
        self.env.rewards = self.rewards

        self.featureExtractor = utils.MapFromDumpExtractor2(self.env)

    def step(self, action):

        ob, reward, done, _ = self.env.step(action)
        change = False

        if ob[self.pos1[0],self.pos1[1]]==2:
            if self.random_generator.uniform(0,1)<=self.probability_transition:
                self.env.rewards = self.rew[0]
            else:
                self.env.rewards = self.rew[1]

            self.stage = 1
            change = True

        if ob[self.pos2[0],self.pos2[1]]==2:
            if self.random_generator.uniform(0,1)<=self.probability_transition:
                self.env.rewards = self.rew[1]
            else:
                self.env.rewards = self.rew[0]

            self.stage = 1
            change = True

        if self.random_generator.uniform(0,1)<=self.switch_rate:
            self.rew.reverse()

        if self.stage == 1 and reward != -0.001 and change==False:

            self.trial +=1
            ob = self.reset()
            self.env.done = False

        if self.trial >= self.trials:
            done = True
            self.env.done = True

        return self.featureExtractor.getFeatures(ob), reward, done, _


    def reset(self):

        self.stage = 0
        self.env.rewards = self.rewards
        ob = self.env.reset()
        return self.featureExtractor.getFeatures(ob)
        #return ob
