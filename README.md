# Reproducibility of "Faster Reinforcement Learning with Expert State Sequences"
OpenReview.net: [Faster Reinforcement Learning with Expert State Sequences](https://openreview.net/forum?id=BJ7d0fW0b)

Our algorithm is implemented in Python3 using Pytorch.

## Code Organization
```
.
├── action_value_estimator.py
├── baseline.py
├── config
│   ├── doom_basic.cfg
│   ├── doom_corridor.cfg
│   ├── doom_defend_center.cfg
│   ├── doom_health_gathering.cfg
│   └── doom_predict_position.cfg
├── data
├── dqn.py
├── experience
├── faster_reinforcement_learning_with_expert_state_sequences.py
├── logs
├── models
├── phi.py
├── README.md
├── setup.sh
├── subgoal_extractor.py
├── utils
│   └── util.py
└── vizdoom.ini
```

* README.md: This README file
* config/: configuration files for each Doom variant used in training and testing.
* data/: data files containing the epoch, simple moving average reward, and current epsilon in csv format.
* docker/: Dockerfile for running the project in a Docker container.
* experience/: expert experience (state, next state) pairs gathered from a pretrained DQN agent.
* figures/: figures used in the workshop paper are stored here.
* logs/: log files.
* models/: pytorch model files trained using the dqn baseline described in the paper being reproduced.
* utils/: utilities for using the ppaquette Doom environments on OpenAI Gym

Run with python3 -o flag to disable assertions

## Installation

1. Clone this repository.

2. Run the following installation procedure:
```
cd iclr_2018_reproducibility_challenge_rmacy_fcyeh
./setup.sh
```

3. Run the training procedure for the baseline expert:
```
python3 baseline.py --config config/doom_basic.cfg --seed 59 --save-data-file data/doom_basic --save-model models/doom_basic
```

4. Run the testing procedure for the baseline expert to generate expert state sequnces:
```
python3 baseline.py --mode test --config/doom_basic.cfg --seed 59 --save-data-file --load-model <models/save_doom_basic.pt> --save-exp experiences/doom_basic
```

5. Run the training procesure for the proposed algorithm using the expert state sequences gathered from the baseline.
```
python3 faster_reinforcement_learning_with_expert_state_sequences.py --config config/doom_basic.cfg --seed 59 --expert-experiences experiences/doom-basic-expert-experiences.p --save-data-file data/doom-basic-frlwess.csv --beta=0.5
```
