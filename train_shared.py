from tools import ConfigManager
config = ConfigManager('shared_model_config.json').config

from train_agent.pre_train_shared import shared_pre_train
import sys

if __name__ == '__main__':   
    from torch.multiprocessing import freeze_support
    freeze_support()
    sys.exit(shared_pre_train(
        config=config,
        vocab_data_path="dataset/minimind_dataset/pretrain_hq.jsonl",
        vocab_trg_path="vocab.json",
        json_data_path="dataset/minimind_dataset/pretrain_hq.jsonl",
        max_len=512,
        batch_size=64,
        max_epochs=1,
        num_workers=8,
        accumulation_steps=1,
        warmup_index = 471,
        keep_temp_index = 3000,
        sava_frq=100,
        last_index=90000,
        n_of_samples=300,
        n_of_samplings=4710,
        print_frq=3,
        conlude_epoch=10,
        seed=42,
        prefetch_factor=6,
        init_lr=5e-4,
        NO_loss_p = True
    ))
