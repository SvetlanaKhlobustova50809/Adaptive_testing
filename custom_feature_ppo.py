'''
Implement the PPO algorithm for our Selection Environment. 
'''
import os
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

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # Modify this part to fit with your input space
        self.feature_extractor = nn.Sequential(nn.Linear(observation_space.shape[0], 128), nn.ReLU())
        # Set the dimensions of the last layer of the feature extractor
        self.latent_dim_pi = 128
        self.latent_dim_vf = 128

    def forward(self, observations):
        return self.feature_extractor(observations), self.feature_extractor(observations)

    def forward_critic(self, observations):
        return self.feature_extractor(observations)

    def forward_actor(self, observations):
        return self.feature_extractor(observations)
    
class RecurrentPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(RecurrentPolicy, self).__init__(*args, **kwargs,
                                           net_arch={'pi': [128, 'lstm', 128],
                                                     'vf': [128, 'lstm', 128]})

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomFeatureExtractor(self.observation_space)        
    

        
def implement_PPO_algorithm(env, num_students, MAX_EPISODES, num_test_students, hyperparameters, save_main_dir = 'results', intermediate=None, bypass_num_passes=None):
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

def save_intermediate_results(env, num_students, MAX_EPISODES, num_test_students, hyperparameters):
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
        implement_PPO_algorithm(env, num_students, MAX_EPISODES, num_test_students, hyperparameters, save_main_dir, intermediate, bypass_num_passes=num_forward_passes)

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
    parser.add_argument('--num_test_students', type=int, default=11, help='Number of test students')
    parser.add_argument('--CONSIDER_TEST_CASES', type=int, default=35, help='Number of test cases to consider')
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

    # test individual environment methods
    # test_individual_env_methods()

    # check environment
    check_env(env)
    print('Environment Validated!')


    if not intermediate:
        # implement algorithm
        print('\nPerforming Hyperparameter Tuning')

        # define hyperparameters for hyperparameter tuning
        policy_name = ['CustomPolicy']
        gamma = [0.9, 1.0]
        ent_coef = [0.01, 0.1, 0.9]
        learning_rate = [0.0001, 0.0005, 0.001, 0.01]
        num_forward_passes = [1, 5, 10] # num epochs
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
            implement_PPO_algorithm(env, len(student_ids), MAX_EPISODES, num_test_students, hyperparameters)
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
        policy_name = 'CustomPolicy'
        hyperparameters = {'policy_name': policy_name, 'gamma': float(gamma), 'ent_coef': float(ent_coef), 'learning_rate': float(learning_rate), 'num_forward_passes': int(num_forward_passes)}

        # create new environment
        env = create_new_env(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)
        print('Saving intermediate results for hyperparameters: ', hyperparameters)
        # save intermediate results
        save_intermediate_results(env, len(student_ids), MAX_EPISODES, num_test_students, hyperparameters)

if __name__ == '__main__':
    main()