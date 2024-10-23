'''
Implement the PPO algorithm for our Selection Environment. 
'''
import os
import sys
import json
import math
import argparse
from tqdm import tqdm
import itertools
from collections import defaultdict
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from IRT.implement_irt import read_dataset
from selection_env import SelectionEnv
import torch
from torch.distributions import Categorical
from functools import partial

class LstmFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=observation_space.shape[0], hidden_size=self.features_dim, num_layers=1, batch_first=True)

    def forward(self, observations):
        # Pass the observations through the LSTM
        lstm_out, _ = self.lstm(observations)

        # Remove the sequence length dimension
        lstm_out = lstm_out.squeeze(1)

        return lstm_out

    
class LstmBilinearPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, test_case_embeddings, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, 
                         features_extractor_class=LstmFeaturesExtractor, 
                         # net_arch=[dict(pi=[64, 64], vf=[64, 64])], 
                         activation_fn=nn.Tanh, 
                         *args, **kwargs)
        

        # TODO: define feature dimension (make it a parameter of this constructor)
        self.features_dim = 128

        # get the test cases embeddings
        self.test_case_embeddings = test_case_embeddings

        # Define the weight matrix W
        self.W = nn.Parameter(torch.randn(self.features_dim, observation_space.shape[0]))

        # Define the softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Define the value function
        self.value_fn = nn.Linear(self.features_dim, 1)

    def _get_action_dist_from_latents(self, latents_pi, latents_vf):

        # compute projections
        projection = torch.matmul(torch.matmul(latents_pi, self.W), self.test_case_embeddings.T)

        # Apply the softmax to get the probabilities of the test cases
        probabilities = self.softmax(projection)

        # Create the action distribution
        action_dist = Categorical(probs=probabilities) # shape: (batch_size, num_test_cases)

        # Compute the value
        value = self.value_fn(latents_vf)

        return action_dist, value

    def forward(self, obs, deterministic=False):
        # Extract the features
        features = self.extract_features(obs)

        # Compute the action distribution and the value
        action_dist, value = self._get_action_dist_from_latents(features, features)

        # Sample an action
        if deterministic:
            action = action_dist.probs.argmax(dim=-1, keepdim=True)
        else:
            action = action_dist.sample()

        # Return the action, the value and the action log probability
        return action, value, action_dist.log_prob(action)
    
def get_test_case_embeddings(env):
    '''
    Get the test case embeddings
    '''
    # get test case dictionary
    test_cases = env.test_cases

    # get test case embeddings
    test_case_embeddings = []
    for case_num, test_case in test_cases.items():
        # disable gradient
        with torch.no_grad():
            # Tokenize the obs str
            obs_input_ids = env.tokenizer.encode(test_case, return_tensors='pt').to(env.device)
            outputs_obs = env.code_gpt(obs_input_ids)
            embeddings_obs = outputs_obs.last_hidden_state
            # Aggregate the embeddings
            embeddings_obs = embeddings_obs.mean(dim=1).squeeze(0)
            # Add the embedding to the list
            test_case_embeddings.append(embeddings_obs)
    # convert to tensor
    test_case_embeddings = torch.stack(test_case_embeddings)
    # return the embeddings
    return test_case_embeddings

     
def implement_PPO_algorithm(env, num_students, MAX_EPISODES, num_test_students, hyperparameters, test_case_embeddings, save_main_dir = 'results', intermediate=None, bypass_num_passes=None):
    '''
    Implement stable baselines algorithm with the environment
    '''
    # wrap the environment using DummyVecEnv
    if intermediate is None:
        obs = env.set_force_reset(force_reset=-1, mode='train') # start from student 0
        env = DummyVecEnv([lambda: env])

    policy_name = hyperparameters['policy_name']
    gamma = hyperparameters['gamma']
    ent_coef = hyperparameters['ent_coef']
    learning_rate = hyperparameters['learning_rate']
    num_forward_passes = hyperparameters['num_forward_passes']
    
    # train students
    num_train_students = num_students - num_test_students

    # define the policy
    RecurrentPolicy = partial(LstmBilinearPolicy, test_case_embeddings=test_case_embeddings)

    # train the agent
    print('\nTraining the agent')
    model = PPO(RecurrentPolicy, env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate, n_epochs = num_forward_passes, n_steps=1024, batch_size=64).learn(total_timesteps = int(num_train_students)*MAX_EPISODES)

    print('\nTesting the trained agent')
    # test the trained agent
    for sub_env in env.envs:
        sub_env.set_force_reset(num_students-num_test_students-1, mode='test')
    obs = env.reset() # start from student 0
    # test for k students
    K = num_test_students-1
    # TODO: Calculate discounted return for each student
    pred_student_information = defaultdict(dict)
    for j in tqdm(range(K)):
        # print('##### Testing for student {:d} #####'.format(j))
        n_steps = 20
        pred_student_information[j]['discounted_return'] = 0
        pred_student_information[j]['test_cases'] = []
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=False)
            # print("Step {}".format(step + 1))
            # print("Action: ", action)
            obs, reward, done, info = env.step(action)
            # print('Observation, Reward, Done, Info: ', obs, reward, done, info)
            env.render(mode='console')
            # update discounted return
            pred_student_information[j]['discounted_return'] += (gamma**step)*reward.tolist()[0]
            # update test cases
            pred_student_information[j]['test_cases'].append(action.tolist()[0])
            if done:
                break
    
    save_dir = '{:s}/PPO/{:s}'.format(save_main_dir, policy_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save discounted return
    if intermediate is None:
        save_path = '{:s}/{:f}_{:f}_{:f}_{:d}.json'.format(save_dir, gamma, ent_coef, learning_rate, num_forward_passes)
        with open(save_path, 'w') as f:
            json.dump(pred_student_information, f, indent=4)
    else:
        save_path_dir = '{:s}/{:f}_{:f}_{:f}_{:d}'.format(save_dir, gamma, ent_coef, learning_rate, bypass_num_passes)
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        save_path = '{:d}.json'.format(intermediate)
        with open(os.path.join(save_path_dir, save_path), 'w') as f:
            json.dump(pred_student_information, f, indent=4)

def save_intermediate_results(env, num_students, MAX_EPISODES, num_test_students, hyperparameters, test_case_embeddings):
    '''
    Train the model and save the validation results after each epoch
    '''
    save_main_dir = 'results_intermediate'
    num_forward_passes = hyperparameters['num_forward_passes']
    hyperparameters['num_forward_passes'] = 1
    for intermediate in range(1, num_forward_passes+1):
        if intermediate == 1:
            # wrap the environment using DummyVecEnv
            obs = env.set_force_reset(force_reset=-1, mode='train') # start from student 0
            env = DummyVecEnv([lambda: env])
        else:
            for sub_env in env.envs:
                sub_env.set_force_reset(force_reset=-1, mode='train')

        # call the implement_PPO_algorithm function
        print('Epoch: {:d}'.format(intermediate))
        implement_PPO_algorithm(env, num_students, MAX_EPISODES, num_test_students, hyperparameters, test_case_embeddings, save_main_dir, intermediate, bypass_num_passes=num_forward_passes)

def test_individual_env_methods():
    env = SelectionEnv()
    observation = env.get_observation('This is a test', 'This is a test also')
    print(observation.shape) # (768,)
    # test get updated ability 
    print(env.get_updated_ability(0, [0, 0, 1, 2]))
    # test - step
    for i in range(10):
        print(env.step(i))

def create_new_env(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose=False):
    '''
    Create a new environment
    '''
    env = SelectionEnv(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)
    return env


def parse_arguments():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Test Case Selection Environment')
    parser.add_argument('--num_test_students', type=int, default=10, help='Number of test students')
    parser.add_argument('--CONSIDER_TEST_CASES', type=int, default=15, help='Number of test cases to consider')
    parser.add_argument('--MAX_EPISODES', type=int, default=10, help='Maximum number of episodes')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
    parser.add_argument('--intermediate', type=bool, default=False, help='Save intermediate results')
    parser.add_argument('--config', type=str, default='0.900000_0.000000_0.005000_10.json', help='Default hyperparameters')
    parser.add_argument('--save_main_dir', type=str, default='results', help='Save main directory')
    parser.add_argument('--force_repeat', type=bool, default=False, help='Force repeat the experiment')
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_arguments()


    # output directory
    output_dir = args.save_main_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define number of test cases to consider
    CONSIDER_TEST_CASES = args.CONSIDER_TEST_CASES
    MAX_EPISODES = args.MAX_EPISODES
    num_test_students = args.num_test_students
    verbose = args.verbose
    intermediate = args.intermediate
    

    # TODO: Load dataset
    student_ids, outputs = read_dataset(CONSIDER_TEST_CASES)

    # define environment
    env = create_new_env(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose)

    # get test case embeddings
    test_case_embeddings = get_test_case_embeddings(env)
    print('Generated Test Case Embeddings: ', test_case_embeddings.shape)

    # test individual environment methods
    # test_individual_env_methods()

    # check environment
    check_env(env)
    print('Environment Validated!')


    if not intermediate:
        # implement algorithm
        print('\nPerforming Hyperparameter Tuning')

        # define hyperparameters for hyperparameter tuning
        policy_name = ['LSTMBilinearPolicy']
        gamma = [0.9]
        ent_coef = [0.01, 0.1]
        learning_rate = [0.0001, 0.001, 0.01]
        num_forward_passes = [10, 25, 50] # num epochs
        # iterate over all combinations of hyperparameters
        for policy_name, gamma, ent_coef, learning_rate, num_forward_passes in itertools.product(policy_name, gamma, ent_coef, learning_rate, num_forward_passes):
            if not args.force_repeat:
                # save path
                save_dir = 'results/PPO/{:s}'.format(policy_name)
                save_path = '{:s}/{:f}_{:f}_{:f}_{:d}.json'.format(save_dir, gamma, ent_coef, learning_rate, num_forward_passes)
                # check if file exists
                if os.path.exists(save_path):
                    continue

            # create new environment
            env = create_new_env(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)
            # define hyperparameters dictionary
            hyperparameters = {'policy_name': policy_name, 'gamma': gamma, 'ent_coef': ent_coef, 'learning_rate': learning_rate, 'num_forward_passes': num_forward_passes}
            # implement algorithm
            implement_PPO_algorithm(env, len(student_ids), MAX_EPISODES, num_test_students, hyperparameters, test_case_embeddings)
            # print hyperparameters
            print('Hyperparameters: ', hyperparameters)
            # close environment
            env.close()
            # delete environment
            del env
    else:
        # print intermediate results for a single hyperparameter setting
        # parse config 
        config = args.config
        gamma, ent_coef, learning_rate, num_forward_passes = config.strip('.json').split('_')
        policy_name = 'LSTMBilinearPolicy'
        hyperparameters = {'policy_name': policy_name, 'gamma': float(gamma), 'ent_coef': float(ent_coef), 'learning_rate': float(learning_rate), 'num_forward_passes': int(num_forward_passes)}

        # create new environment
        env = create_new_env(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)
        print('Saving intermediate results for hyperparameters: ', hyperparameters)
        # save intermediate results
        save_intermediate_results(env, len(student_ids), MAX_EPISODES, num_test_students, hyperparameters, test_case_embeddings)

if __name__ == '__main__':
    main()