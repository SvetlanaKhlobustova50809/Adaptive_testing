import os
import json
import math
import argparse
from tqdm import tqdm
import itertools
from collections import defaultdict
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from IRT.implement_irt import read_dataset
from selection_env import SelectionEnv


def implement_PPO_algorithm(env, num_students, MAX_EPISODES, num_test_students, hyperparameters, save_main_dir='results', intermediate=None, bypass_num_passes=None):
    # wrap the environment using DummyVecEnv
    if intermediate is None:
        obs = env.set_force_reset(
            force_reset=-1, mode='train') 
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
    model = PPO(policy_name, env, verbose=1, gamma=gamma, ent_coef=ent_coef, learning_rate=learning_rate,
                n_epochs=num_forward_passes, n_steps=1024, batch_size=64).learn(total_timesteps=int(num_train_students)*MAX_EPISODES)

    print('\nTesting the trained agent')
    # test the trained agent
    for sub_env in env.envs:
        sub_env.set_force_reset(num_students-num_test_students-1, mode='test')
    obs = env.reset()  # start from student 0

    K = num_test_students
    # TODO: Calculate discounted return for each student
    pred_student_information = defaultdict(dict)
    for j in tqdm(range(1, K)):
        print('##### Testing for student {:d} #####'.format(j))
        n_steps = 40
        pred_student_information[j]['discounted_return'] = 0
        pred_student_information[j]['test_cases'] = []
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=False)
            # print("Step {}".format(step + 1))
            print("Action: ", action)
            obs, reward, done, info = env.step(action)
            print("Reward: ", reward)
            # print('Observation, Reward, Done, Info: ', obs, reward, done, info)
            env.render(mode='console')

            pred_student_information[j]['discounted_return'] += (
                gamma**step)*reward.tolist()[0]
            pred_student_information[j]['test_cases'].append(
                action.tolist()[0])
            if done:
                break

    save_dir = '{:s}/PPO/{:s}'.format(save_main_dir, policy_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save discounted return
    if intermediate is None:
        save_path = '{:s}/{:f}_{:f}_{:f}_{:d}.json'.format(
            save_dir, gamma, ent_coef, learning_rate, num_forward_passes)
        with open(save_path, 'w') as f:
            json.dump(pred_student_information, f, indent=4)
    else:
        save_path_dir = '{:s}/{:f}_{:f}_{:f}_{:d}'.format(
            save_dir, gamma, ent_coef, learning_rate, bypass_num_passes)
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
            obs = env.set_force_reset(
                force_reset=-1, mode='train')  # start from student 0
            env = DummyVecEnv([lambda: env])
        else:
            for sub_env in env.envs:
                sub_env.set_force_reset(force_reset=-1, mode='train')

        # call the implement_PPO_algorithm function
        print('Epoch: {:d}'.format(intermediate))
        implement_PPO_algorithm(env, num_students, MAX_EPISODES, num_test_students,
                                hyperparameters, save_main_dir, intermediate, bypass_num_passes=num_forward_passes)


def test_individual_env_methods():
    env = SelectionEnv()
    observation = env.get_observation('This is a test', 'This is a test also')
    # print(observation.shape) # (768,)
    # test get updated ability
    #print(env.get_updated_ability(0, [0, 0, 1, 2]))
    # test - step
    for i in range(10):
        print(env.step(i))


def create_new_env(student_ids, num_test_students, outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose=False):
    '''
    Create a new environment
    '''
    env = SelectionEnv(student_ids, num_test_students, outputs,
                       CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)
    return env


def parse_arguments():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Test Case Selection Environment')
    parser.add_argument('--num_test_students', type=int,
                        default=11, help='Number of test students')
    parser.add_argument('--CONSIDER_TEST_CASES', type=int,
                        default=35, help='Number of test cases to consider')
    parser.add_argument('--MAX_EPISODES', type=int,
                        default=10, help='Maximum number of episodes')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
    parser.add_argument('--intermediate', type=bool,
                        default=False, help='Save intermediate results')
    parser.add_argument(
        '--config', type=str, default='0.900000_0.000000_0.005000_10.json', help='Default hyperparameters')
    parser.add_argument('--save_main_dir', type=str,
                        default='results', help='Save main directory')
    parser.add_argument('--force_repeat', type=bool,
                        default=False, help='Force repeat the experiment')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    output_dir = args.save_main_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    CONSIDER_TEST_CASES = args.CONSIDER_TEST_CASES
    MAX_EPISODES = args.MAX_EPISODES
    num_test_students = args.num_test_students
    verbose = args.verbose
    intermediate = args.intermediate

    student_ids, outputs = read_dataset(CONSIDER_TEST_CASES)

    env = create_new_env(student_ids, num_test_students,
                         outputs, CONSIDER_TEST_CASES, MAX_EPISODES, verbose)

    check_env(env)
    print('Environment Validated!')

    if not intermediate:
        # implement algorithm
        print('\nPerforming Hyperparameter Tuning')

        # define hyperparameters for hyperparameter tuning
        policy_name = ['MlpPolicy']
        gamma = [0.9]
        ent_coef = [0.01, 0.1]
        learning_rate = [0.0001, 0.001, 0.01]
        num_forward_passes = [10, 25, 50]  # num epochs

        for policy_name, gamma, ent_coef, learning_rate, num_forward_passes in itertools.product(policy_name, gamma, ent_coef, learning_rate, num_forward_passes):
            if not args.force_repeat:
                save_dir = 'results/PPO/{:s}'.format(policy_name)
                save_path = save_path = '{:s}/{:f}_{:f}_{:f}_{:d}.json'.format(
                    save_dir, gamma, ent_coef, learning_rate, num_forward_passes)

                if os.path.exists(save_path):
                    continue

            env = create_new_env(student_ids, num_test_students, outputs,
                                 CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)

            hyperparameters = {'policy_name': policy_name, 'gamma': gamma, 'ent_coef': ent_coef,
                               'learning_rate': learning_rate, 'num_forward_passes': num_forward_passes}

            implement_PPO_algorithm(
                env, len(student_ids), MAX_EPISODES, num_test_students, hyperparameters)

            print('Hyperparameters: ', hyperparameters)

            env.close()

            del env
    else:
        config = args.config
        gamma, ent_coef, learning_rate, num_forward_passes = config.strip(
            '.json').split('_')
        policy_name = 'MlpPolicy'
        hyperparameters = {'policy_name': policy_name, 'gamma': float(gamma), 'ent_coef': float(
            ent_coef), 'learning_rate': float(learning_rate), 'num_forward_passes': int(num_forward_passes)}

        env = create_new_env(student_ids, num_test_students, outputs,
                             CONSIDER_TEST_CASES, MAX_EPISODES, verbose=verbose)
        print('Saving intermediate results for hyperparameters: ', hyperparameters)

        save_intermediate_results(
            env, len(student_ids), MAX_EPISODES, num_test_students, hyperparameters)


if __name__ == '__main__':
    main()
