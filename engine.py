"""
Phonon 2.5 Runtime Engine
Extreme 2-bit quantization and real-time streaming inference for FastConformer RNN-T
Author: Ramin Mahmoodi (https://github.com/ramin-mahmoodi)
License: Apache-2.0
"""

import os
import json
import torch
import numpy as np
import zstandard as zstd
from transformers import AutoModelForRNNT, AutoProcessor

GROUP_SIZE = 128


def unpack_five_value_g128(blob: bytes, shape: list, group_size: int = 128) -> np.ndarray:
    """
    Unpacks Five-Value 2-bit Base-3 trits and bitplanes into FP16 weights.
    """
    total = int(np.prod(shape))
    n_groups = total // group_size
    rb = (total + 4) // 5
    trits_b = blob[:rb]
    off = rb
    x = np.frombuffer(trits_b, dtype=np.uint8).astype(np.uint16)
    d = np.empty((len(x), 5), dtype=np.uint8)
    for sk in range(5):
        d[:, sk] = (x // (3 ** sk)) % 3
    u_codes = d.ravel()[:total]
    u_nzmask = (u_codes != 1)
    nnz = int(u_nzmask.sum())
    rbytes = (nnz + 7) // 8
    bits_b = blob[off : off + rbytes]
    off += rbytes
    s_lo = np.frombuffer(blob[off : off + 2 * n_groups], dtype=np.float16).astype(np.float32)
    s_hi = np.frombuffer(blob[off + 2 * n_groups : off + 4 * n_groups], dtype=np.float16).astype(np.float32)
    u_bits = np.unpackbits(np.frombuffer(bits_b, dtype=np.uint8), bitorder="little")[:nnz].astype(bool)
    u_is_hi = np.zeros(total, dtype=bool)
    u_is_hi[u_nzmask] = u_bits
    u_mag = np.where(u_is_hi.reshape(n_groups, group_size), s_hi[:, None], s_lo[:, None])
    u_sign = u_codes.reshape(n_groups, group_size).astype(np.int8) - 1
    return (u_sign * u_mag).reshape(shape).astype(np.float16)


def unpack_int6_row(blob: bytes, shape: list) -> np.ndarray:
    """
    Unpacks row-wise scaled INT6 representations into FP16 weights.
    """
    o = shape[0]
    total = int(np.prod(shape))
    ncol = total // o
    scales = np.frombuffer(blob[: 2 * o], dtype=np.float16)
    raw_bytes = np.frombuffer(blob[2 * o :], dtype=np.uint8)
    n_blocks = len(raw_bytes) // 3
    blocks = raw_bytes[: n_blocks * 3].reshape(n_blocks, 3)
    p24 = blocks[:, 0].astype(np.uint32) | (blocks[:, 1].astype(np.uint32) << 8) | (blocks[:, 2].astype(np.uint32) << 16)
    q_unsigned = np.column_stack([(p24 >> s) & 0x3F for s in (0, 6, 12, 18)]).ravel()
    q = q_unsigned[:total].astype(np.int32) - 32
    return (q.reshape(o, ncol).astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(shape).astype(np.float16)


def load_phonon_model(
    container_path: str = None,
    repo_id: str = "ramin-mahmoodi/phonon-nemotron-0.6b-farsi-2bit",
    device: str = None,
):
    """
    Loads and decompresses the Phonon 2.5 container into an active PyTorch RNN-T model.
    If container_path is None or does not exist, automatically downloads from Hugging Face Hub.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if container_path is None or not os.path.exists(container_path):
        from huggingface_hub import hf_hub_download
        print(f"📥 Downloading container from Hugging Face Hub ({repo_id})...")
        container_path = hf_hub_download(
            repo_id=repo_id,
            filename="nemotron_farsi_phonon2.5_step5000_sub200mb.bps.tar.zst",
        )

    print(f"🔓 Decompressing Phonon 2.5 container from: {container_path}")
    dctx = zstd.ZstdDecompressor()
    with open(container_path, "rb") as f_in:
        with dctx.stream_reader(f_in) as reader:
            decompressed_bytes = reader.read()

    m_len = int.from_bytes(decompressed_bytes[:8], "little")
    manifest = json.loads(decompressed_bytes[8 : 8 + m_len].decode("utf-8"))
    data_store = decompressed_bytes[8 + m_len :]

    base_model_id = "mehdi-hf/nemotron-asr-streaming-farsi"
    print(f"🚀 Initializing base architecture from: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.set_num_lookahead_tokens(13)

    model = AutoModelForRNNT.from_pretrained(base_model_id, torch_dtype=torch.float16).to(device)
    model_dict = model.state_dict()

    print(f"⚡ Injecting {len(manifest['records'])} quantized & protected layers into model...")
    for rec in manifest["records"]:
        name = rec["name"]
        blob = data_store[rec["offset"] : rec["offset"] + rec["length"]]
        shape = rec["shape"]
        dtype = rec["dtype"]
        if dtype == "five_value":
            arr = unpack_five_value_g128(blob, shape, GROUP_SIZE)
        elif dtype == "int6":
            arr = unpack_int6_row(blob, shape)
        else:
            arr = np.frombuffer(blob, dtype=np.float16).reshape(shape)
        model_dict[name].copy_(torch.from_numpy(arr).to(device))

    model.load_state_dict(model_dict)
    model.eval()
    print("✅ Model successfully initialized and ready for streaming inference!")
    return model, processor
