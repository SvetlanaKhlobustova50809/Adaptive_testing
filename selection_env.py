import os
import json
import math
import argparse
from tqdm import tqdm
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import DQN, A2C, DDPG, TD3, SAC
import transformers
from transformers import GPT2Tokenizer, GPT2Model
from utils import get_code_dict, get_problem_statement, load_test_cases, get_pivot_code_id
from IRT.implement_irt import read_dataset, IRTModel, get_dataloader, set_seed, get_model_info, train_IRT
from IRT.load_params import load_irt_parameters

class SelectionEnv(gym.Env):    

    def __init__(self, student_ids, num_test_students, outputs, CONSIDER_TEST_CASES=15, MAX_EPISODES=5, verbose=False):
        super(SelectionEnv, self).__init__()

        self.CONSIDER_TEST_CASES = CONSIDER_TEST_CASES
        self.MAX_EPISODES = MAX_EPISODES
        self.verbose = verbose

        self.student_ids = student_ids
        self.num_test_students = num_test_students
        self.outputs = outputs

        # Load IRT parameters
        self.student_ability, self.item_difficulty = load_irt_parameters()
        
        self.code_df = pd.read_csv('IRT_dataset/CodeStates.csv')
        self.pivot_code_id = get_pivot_code_id()
        self.code_dict = get_code_dict(self.student_ids, self.pivot_code_id, self.code_df)
        self.problem_statement = get_problem_statement()
        self.pivot_code = self.code_dict[self.pivot_code_id]
        self.test_cases = dict(itertools.islice(load_test_cases().items(), self.CONSIDER_TEST_CASES))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/CodeGPT-small-java")
        # model - CodeGPT
        self.code_gpt = GPT2Model.from_pretrained("microsoft/CodeGPT-small-java").to(self.device)

        self.student_tracker = -1
        self.episode = 0
        self.max_episodes = self.MAX_EPISODES
        self.test_cases_per_student = defaultdict(list)
        self.ability_per_student = defaultdict(list)
 
        self.mode = 'train'
        self.action_space = spaces.Discrete(len(self.test_cases))

        # observation space - output of code gpt
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
    
    def construct_str_state(self, cur_test_cases):
        # dictionary containing the chosen test case index and the test case value 
        test_case_dict = {i: self.test_cases[str(i)] for i in cur_test_cases}

        str_state = 'Program: {:s}\t Student Code: {:s}\tChosen Test Cases: {:s}'.format(self.problem_statement, self.code_dict[self.student_ids[self.student_tracker]], str(test_case_dict))

        correct_code_str = 'Program: {:s}\t Correct Code: {:s}\tChosen Test Cases: {:s}'.format(self.problem_statement, self.pivot_code, str(test_case_dict))
        # print(str_state)
        return str_state, correct_code_str

    def get_observation(self, obs_str, pivot_code_str):
        '''
        Pass the observation string through the code bert with gradient disabled
        '''
        # disable gradient
        with torch.no_grad():
            # Tokenize the obs str
            obs_input_ids = self.tokenizer.encode(obs_str, return_tensors='pt').to(self.device)
            outputs_obs = self.code_gpt(obs_input_ids)
            embeddings_obs = outputs_obs.last_hidden_state
            # Aggregate the embeddings
            embeddings_obs = embeddings_obs.mean(dim=1).squeeze(0)

            # Tokenize the pivot code str
            pivot_input_ids = self.tokenizer.encode(pivot_code_str, return_tensors='pt').to(self.device)
            # for pivot code string
            outputs_pivot = self.code_gpt(pivot_input_ids)
            embeddings_pivot = outputs_pivot.last_hidden_state
            # Aggregate the embeddings
            embeddings_pivot = embeddings_pivot.mean(dim=1).squeeze(0)

            # state representation (subtract the two)
            state_rep = embeddings_obs - embeddings_pivot
            # convert to numpy array 
            state_rep = state_rep.cpu().numpy()
            
            return state_rep
    
    def set_force_reset(self, force_reset=0, mode='train'):
        self.student_tracker = force_reset
        self.mode = mode
  
    def reset(self, seed=None):
        '''
        Reset the environment
        '''
        # Set the seed if provided
        if seed is not None:
            self.seed(seed)
            # set seed
            set_seed(seed)
        # increment student tracker
        self.student_tracker += 1
        
        # check if student tracker has reached the end
        if self.student_tracker >= len(self.student_ids)-self.num_test_students and self.mode == 'train':
            self.student_tracker = 0
        # print('Student Tracker: {:d}'.format(self.student_tracker))        

        # reset number of episodes
        self.episode = 0
        # reset test cases per student
        self.test_cases_per_student = defaultdict(list)
        # reset ability per student
        self.ability_per_student = defaultdict(list)

        # observation string
        str_state, correct_code_str = self.construct_str_state(self.test_cases_per_student[self.student_tracker])
        # observation 
        observation = self.get_observation(str_state, correct_code_str)

        # info
        info = {'Message': 'Student {:s} is selected'.format(self.student_ids[self.student_tracker])}
        # return observation and info
        return observation, info

    def get_updated_ability(self, student_id, test_cases):
        '''
        Get updated ability of the student using IRT model
        '''
        #get output values of the test cases 
        output_values = []
        for test_case in test_cases:
            output_values.append(self.outputs[student_id][test_case])
        # get the dataloader
        dataloader = get_dataloader(1, [0], [output_values]) # student id is 0, since we are only considering one student at a time
        # get model info
        model, loss_fn, optimizer, num_epochs, device = get_model_info(1, len(test_cases), load_params=True, verbose=False)
        # train the model
        model = train_IRT(test_cases, model, loss_fn, optimizer, num_epochs, device, dataloader, verbose=False)
        # get updated ability
        updated_ability = model.student_ability.item()
        # return updated ability
        return updated_ability

    def step(self, action):
        '''
        Take a step in the environemnt
        '''
        if self.verbose:
            print('Student Tracker: {:d}, Action: {}, Test Cases Per Student: {}'.format(self.student_tracker, action, self.test_cases_per_student[self.student_tracker]))
        # increment episode
        self.episode += 1
        # update chosen test cases list
        self.test_cases_per_student[self.student_tracker].append(action)
        # update student ability using IRT model
        updated_ability = self.get_updated_ability(self.student_tracker, self.test_cases_per_student[self.student_tracker])
        # update ability per student
        self.ability_per_student[self.student_tracker].append(updated_ability)
        # compute reward 
        # check if action has been selected previously 
        count_action = self.test_cases_per_student[self.student_tracker].count(action)
        if count_action > 1:
            # penalize the agent for selecting the same action again
            reward = -10000
        else:
            try:
                reward = 1/abs(self.student_ability[self.student_tracker].item() - updated_ability)
            except ZeroDivisionError:
                reward = 10000
        # update next state
        str_state, correct_code_str = self.construct_str_state(self.test_cases_per_student[self.student_tracker]) 
        # update next state
        next_state = self.get_observation(str_state, correct_code_str)
        # check termination of episode
        done = False
        if self.episode == self.max_episodes:
            done = True
        # set truncate and info
        truncate = False
        info = {}

        # return information 
        return next_state, reward, done, truncate, info

    def render(self, mode='console'):
        if mode == 'console':
            print('Episode: {:d}\tStudent: {:s}\tTest Cases: {:s}\tStudent Ability: {:.3f}'.format(self.episode, self.student_ids[self.student_tracker], str(self.test_cases_per_student[self.student_tracker]), self.ability_per_student[self.student_tracker][-1]))
        else:
            pass
    
    def close(self):
        pass

    def seed(self, seed=None):
        set_seed(seed)

