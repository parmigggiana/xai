from src.finetuning import FinetuneConfig, run_finetuning

DATASET_NAMES = ["CHAOS", "MMWHS"]
DOMAINS = ["CT", "MR"]
DATA_PATH = "data/"
CHECKPOINT_PATH = "checkpoints/"
OUTPUTS_PATH = "outputs/"
USE_3D = True
TRAINING_EPOCHS = {
    ("CHAOS", "CT"): 1,
    ("CHAOS", "MR"): 1,
    ("MMWHS", "CT"): 1,
    ("MMWHS", "MR"): 1,
}
BATCH_SIZE = 4
SPATIAL_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
SAVE_BEST = True
EARLY_STOP_PATIENCE = 5
VAL_MAX_BATCHES = None
# Number of DataLoader workers (set >0 to enable parallel data loading)
NUM_WORKERS = 0
# Set True to enable debug prints/timers/visualizations)
DEBUG = False
MEMORY_TRACE = False

# Profiling controls: False | 'cprofile' | 'torch'
PROFILE = False
if __name__ == "__main__":
    config = FinetuneConfig(
        dataset_names=DATASET_NAMES,
        domains=DOMAINS,
        training_epochs=TRAINING_EPOCHS,
        data_path=DATA_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        outputs_path=OUTPUTS_PATH,
        use_3d=USE_3D,
        batch_size=BATCH_SIZE,
        spatial_size=SPATIAL_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_best=SAVE_BEST,
        early_stop_patience=EARLY_STOP_PATIENCE,
        val_max_batches=VAL_MAX_BATCHES,
        num_workers=NUM_WORKERS,
        debug=DEBUG,
        memory_trace=MEMORY_TRACE,
        profile=PROFILE,
    )
    run_finetuning(config)
