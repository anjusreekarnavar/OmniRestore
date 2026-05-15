**OmniRestore: Robust Universal Image Restoration from Combined and Unspecified Degradations**











For setting up the environment

pip install -r requirements.txt


Train the baseline module

Combined training with shared-encoder and five decoders 

python -m torch.distributed.launch --nproc_per_node node_vlaue pretrain_baseline.py --train_data_path path --val_data_path path --log_dir path   --output_dir path  --decoder_depth depthvalue --log_dir path

For using multigpus include --nproc_per_node=num_gpus, change the num_gpus as per the requirement

For running the aggregator module  use

python -m torch.distributed.launch --nproc_per_node num_gpus --master_port8871 pretrain_moe.py --output_dir path --log_dir path --train_dir path --val_dir path


## Code Structure

DeepLearningProject/
│
├── datasets/                # Datasets for training and testing
│   ├── raw/                 # Raw datasets (unprocessed)
│   └── processed/           # Preprocessed datasets (e.g., split, normalized, etc.)
│
├── models/                  # Store model architectures and pre-trained weights
│   ├── architectures/       # Different model definitions (e.g., encoder, decoder, etc.)
│   └── checkpoints/         # Saved model checkpoints for resuming or evaluation
│
├── notebooks/               # Jupyter notebooks for experiments, exploration, etc.
│
├── src/                     # Core source code for training, testing, and utilities
│   ├── data/                # Data loaders and preprocessing scripts
│   ├── utils/               # Utility functions (e.g., metrics, visualization)
│   ├── training/            # Training scripts and loops
│   ├── testing/             # Testing and evaluation scripts
│   └── config/              # Configuration files for experiments
│
├── results/                 # Output results (metrics, figures, logs)
│   ├── logs/                # Training logs
│   ├── plots/               # Generated plots (e.g., loss, accuracy curves)
│   └── predictions/         # Model output (e.g., predicted images)
│
├── scripts/                 # Scripts to run the pipeline (train, test, preprocess, etc.)
│   └── run_experiment.sh    # Example bash script to run training or testing
│
├── tests/                   # Unit and integration tests
│
├── docs/                    # Documentation (Sphinx or other tools)
│
├── requirements.txt         # List of dependencies
├── README.md                # Overview of the project
├── LICENSE                  # License for the project
└── .gitignore               # Git ignore file
