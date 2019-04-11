# model config

fs = 16000
frame_len = 512
frame_hop = 256

nnet_conf = {
    "frame_len": frame_len,
    "frame_hop": frame_hop,
    "round_pow_of_two": True,
    "embedding_dim": 256,
    "non_linear": "relu"
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "logging_period": 200,
    "gradient_clip": 10,
    "min_lr": 1e-8,
    "patience": 1,
    "factor": 0.5
}

train_dir = "data/train/"
dev_dir = "data/dev/"

train_data = {
    "sr": fs,
    "mix_scp": train_dir + "mix.scp",
    "ref_scp": train_dir + "ref.scp",
    "emb_scp": train_dir + "emb.scp"
}

dev_data = {
    "sr": fs,
    "mix_scp": dev_dir + "mix.scp",
    "ref_scp": dev_dir + "ref.scp",
    "emb_scp": dev_dir + "emb.scp"
}
