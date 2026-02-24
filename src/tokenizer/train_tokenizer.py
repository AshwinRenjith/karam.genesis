# src/tokenizer/train_tokenizer.py
"""
Train a 32K SentencePiece BPE tokenizer on FineWeb-Edu.
Streams 1M sentences — never writes full dataset to disk.
Runtime: ~2 hours on M1.
"""
import io
import sentencepiece as spm
from datasets import load_dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VOCAB_SIZE = 32_000
OUTPUT_DIR = Path("checkpoints/tokenizer")
MODEL_PREFIX = str(OUTPUT_DIR / "karam_spm_32k")
N_SENTENCES = 1_000_000


def stream_fineweb_sentences(n: int) -> io.StringIO:
    """Stream n sentences from FineWeb-Edu into an in-memory buffer."""
    log.info(f"Streaming {n:,} sentences from FineWeb-Edu...")
    buf = io.StringIO()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    count = 0
    for sample in dataset:
        # Each sample is a document — take first 3 sentences per doc
        text = sample["text"]
        sentences = text.split(". ")[:3]
        for s in sentences:
            s = s.strip()
            if len(s) > 20:  # Filter noise
                buf.write(s + "\n")
                count += 1
                if count >= n:
                    break
        if count >= n:
            break
    log.info(f"Streamed {count:,} sentences.")
    buf.seek(0)
    return buf


def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write sentences to a temp file (spm requires a file path)
    tmp_path = OUTPUT_DIR / "train_corpus.txt"
    buf = stream_fineweb_sentences(N_SENTENCES)
    tmp_path.write_text(buf.read())
    log.info(f"Corpus written to {tmp_path}")

    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<mask>"],  # id=4
        num_threads=8,
        input_sentence_size=N_SENTENCES,
        shuffle_input_sentence=True,
    )
    log.info(f"Tokenizer saved: {MODEL_PREFIX}.model")

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{MODEL_PREFIX}.model")
    test = "The quick brown fox jumps over the lazy dog."
    tokens = sp.Encode(test, out_type=str)
    ids = sp.Encode(test)
    log.info(f"Test encode: {tokens}")
    log.info(f"Test IDs: {ids}")
    log.info(f"Vocab size confirmed: {sp.GetPieceSize()}")
    assert sp.GetPieceSize() == VOCAB_SIZE

    # Cleanup temp corpus
    tmp_path.unlink()
    log.info("✅ Tokenizer training complete.")


if __name__ == "__main__":
    train()
