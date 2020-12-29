#File where we define any necessary tasks for agent to learn
import random

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

