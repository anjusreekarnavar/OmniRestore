Installation   

pip install -r requirements.txt


Train the baseline model

Training with single-encoder -single decoder (For independently training five experts)

python -m torch.distributed.launch pretrain_decoder.py --train_data_path path --val_data_path path --log_dir path   --output_dir path  --decoder_depth value --blr value

Training with shared encoder (for combined training)

python -m torch.distributed.launch shared_encoder_pretrain.py --train_data_path path --val_data_path path --log_dir path   --output_dir path  


for using multigpus include --nproc_per_node=num_gpus


For running aggregator use

python -m torch.distributed.launch moe_trainddp.py


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
