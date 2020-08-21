from agent import DQN, DDQN

# Environment list
envs = ['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']


# CartPole hyper parameters
CartPole_HYPER_PARAMETERS = {'MEM_REPLAY_SIZE': 150000,
                                'BATCH_SIZE': 128,
                                'GAMMA': 0.999,
                                'EPS_START': 0.95,
                                'EPS_END': 0.1,
                                'EPS_DECAY': 1000,
                                'EVALUATE_FREQUENCY': 20,
                                'ALTER_TARGET_UPDATE_RATE': 0.995,
                                'MAX_EPISODES': 1000}


# MountainCar hyper parameters
MountainCar_HYPER_PARAMETERS = {'MEM_REPLAY_SIZE': 150000,
                                'BATCH_SIZE': 512,
                                'GAMMA': 0.999,
                                'EPS_START': 1,
                                'EPS_END': 0.1,
                                'EPS_DECAY': 1000,
                                'EVALUATE_FREQUENCY': 1,
                                'ALTER_TARGET_UPDATE_RATE': 0.999,
                                'MAX_EPISODES': 1000}


# Acrobot hyper parameters
Acrobot_HYPER_PARAMETERS = {'MEM_REPLAY_SIZE': 150000,
                                'BATCH_SIZE': 128,
                                'GAMMA': 0.999,
                                'EPS_START': 1,
                                'EPS_END': 0.1,
                                'EPS_DECAY': 1000,
                                'EVALUATE_FREQUENCY': 20,
                                'ALTER_TARGET_UPDATE_RATE': 0.995,
                                'MAX_EPISODES': 1000}


DQN.train_model(MountainCar_HYPER_PARAMETERS, envs[2])
DDQN.train_model(MountainCar_HYPER_PARAMETERS, envs[2])
