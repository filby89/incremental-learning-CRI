# incremental-learning-CRI
This repository holds the source code for the modules of the paper "Visual Robotic Perception System with Incremental Learning for Child-Robot Interaction Scenarios".


### Emotion Recognition Module
To train the emotion recognition module run:

``
python train_children_tsn.py -c EmoReact/config.json --categorical --continuous --num_segments 3 --exp_name "3 Flow" --arch resnet50 --batch-size 8 --lr 1e-2 --modality Flow -d 1
``

selecting your preferred number of segments and modality.


### Action Recognition Module
To train the action recognition module run:

``
python train_children_tsn_action.py -c BabyAction/config.json --num_segments 5 --exp_name "5 RGB" --arch resnet50 --batch-size 8 --lr 1e-4 --modality RGB -d 2
``


selecting your preferred number of segments and modality.



### Incremental Learning
The incremental learning module is inside the ``FACIL`` directory (initial code from https://github.com/mmasana/FACIL). Example of running the extended icarl for 5 exemplars-per-class and 10 tasks:

``
python main_incremental_babyaction.py --approach icarl --num-exemplars-per-class 5 --num-tasks 10 --nepochs 60 --lr-patience 100
``
