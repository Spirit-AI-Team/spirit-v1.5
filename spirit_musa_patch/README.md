# Spirit-v1.5 MUSA FSDP2 Fine-tuning

The MUSA runtime uses the repository's bundled `checkpoint_config/` and
`fake_data/move_objects_into_box/` by default. From the repository root, start
the validated 8-card fine-tuning configuration with:

```bash
bash spirit_musa_patch/scripts/run_finetune_fsdp2.sh
```

The launcher uses `.venv/bin/python`, 8 visible MUSA devices, batch size 320,
gradient accumulation 4, and 50 micro-steps by default. It invokes
`spirit_musa_patch/train_fsdp2.py` directly through `torch.distributed.run`; the
former `scripts/run_fsdp2.py` wrapper is no longer needed. Common overrides are:

```bash
SPIRIT_BATCH_SIZE=1 \
SPIRIT_GRAD_ACCUM_STEPS=1 \
SPIRIT_MAX_TRAIN_STEPS=1 \
NPROC_PER_NODE=2 \
MUSA_VISIBLE_DEVICES=0,1 \
bash spirit_musa_patch/scripts/run_finetune_fsdp2.sh
```

To use another dataset, checkpoint, or output directory, override
`DATA_ROOT`, `SPIRIT_PRETRAINED_PATH`, or `SPIRIT_OUTPUT_DIR`. The checkpoint
directory must contain `config.json` and `model.safetensors`.

Distributed launch settings are supplied through environment variables. The
default is a single-node run using `127.0.0.1`. For a multi-node run, set the
same `MASTER_ADDR` and `MASTER_PORT` on every node, and set `NNODES`,
`NODE_RANK`, and `MCCL_IB_HCA` for each node. `MCCL_SOCKET_IFNAME` is optional
when the transport requires an explicit network interface. For example:

```bash
# rank 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank-0-address> MCCL_IB_HCA=<rank-0-hca> \
MCCL_SOCKET_IFNAME=<network-interface> \
bash spirit_musa_patch/scripts/run_finetune_fsdp2.sh

# rank 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=<rank-0-address> MCCL_IB_HCA=<rank-1-hca> \
MCCL_SOCKET_IFNAME=<network-interface> \
bash spirit_musa_patch/scripts/run_finetune_fsdp2.sh
```

The merged training entrypoint also accepts the previous command-line options,
but launcher defaults can be supplied through the corresponding environment
variables. In particular, `SPIRIT_RESUME_TRAINING_STATE` still validates the
saved micro-step and resumes from the remaining step count.
