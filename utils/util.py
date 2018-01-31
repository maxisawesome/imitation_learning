import gym
from gym.core import ObservationWrapper
from gym.spaces.box import Box
from scipy.misc import imresize
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'subgoal', 'action', 'next_state', 'next_subgoal', 'reward'))

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=80, width=80, grayscale=True,
                 crop=lambda img: img):
        """ A gym wrapper that crops, scales image into desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1)) # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        img = np.squeeze(img)
        return img

def make_env(arg_env_spec, arg_env_spec_id):
    env_spec = gym.spec(arg_env_spec)
    env_spec.id = arg_env_spec_id
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                                 width=80, height=80, grayscale=True)
    return e

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] 
        self.position = 0
      
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
