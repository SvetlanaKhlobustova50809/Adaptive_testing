# SB3-tutorial

This repository contains code for the tutorial on using Stable Baselines 3 for creating custom environments and custom policies. A blog on the problem statement and the MDP formulation can be found at - https://nish-19.github.io/posts/2023/12/blog-post-6/. 

## Contents 

1. [Installation](#installation) 
2. [Custom Environment](#custom-environment)
3. [PPO](#ppo)
4. [Custom Feature Extractor](#custom-feature-extractor)
5. [Custom Policy (LSTM Bilinear)](#custom-policy)

## Installation

To install the python libraries using ```conda``` execute the following command: 

```
conda env create -f environment.yml
```
## Custom Environment

```selection_env.py``` contains the code for our custom environment. The ```SelectionEnv``` class implements the custom environment and it extends from the OpenAI Gymnasium Environment ```gymnasium.Env```

The method ```reset```  is used for resetting the environment and initializing the state. The method ```step``` executes an action in the current state and returns the next state, reward, and an indication whether the episode is completed or not. 

## PPO

```
python ppo.py
```

The function ```implement_PPO_algorithm``` implements the training and the testing procedure of the PPO algorithm. 

## Custom Feature Extractor 

```
python custom_feature_ppo.py
```

```CustomFeatureExtractor``` class implements a custom feature extraction layer containing 128 hidden units. ```RecurrentPolicy``` implements an LSTM over the features extracted from the state space using ```CustomFeatureExtractor``` class.

## Custom Policy

```
python lstm_bilinear_policy.py
```

```LstmBilinearPolicy``` implements a custom policy which uses an LSTM to extract features from the state representation using the ```LstmFeaturesExtractor``` class. The custom policy learns a projecition from the output of the LSTM to the space of the test cases represented using the test case embeddings (using a Transformer model). The methods ```_get_action_dist_from_latents``` and ```forward``` need to be overridden for implementing this feature. 