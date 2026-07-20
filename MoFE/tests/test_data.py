from __future__ import annotations

import argparse
import tempfile
import unittest
from itertools import islice
from pathlib import Path

from datasets import Dataset, IterableDataset

from MoFE.train import (
    load_training_dataset,
    resolve_local_dataset,
    training_dataloader_num_workers,
)


class TinyTokenizer:
    def __call__(
        self, texts: list[str], *, add_special_tokens: bool
    ) -> dict[str, list[list[int]]]:
        if add_special_tokens:
            raise AssertionError("training tokenization must not add special tokens")
        input_ids = [
            [(ord(character) % 31) + 1 for character in text] for text in texts
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(values) for values in input_ids],
        }


def streaming_args(path: Path, seed: int = 17) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name=None,
        dataset_config=None,
        train_split="train",
        train_file=str(path),
        text_column="text",
        block_size=4,
        streaming=False,
        shuffle_buffer_size=2,
        preprocessing_batch_size=2,
        seed=seed,
    )


class TrainingDataTest(unittest.TestCase):
    def test_parquet_directory_is_sorted_and_streamed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Dataset.from_dict({"text": ["bbbbbbbb", "cccccccc"]}).to_parquet(
                root / "001.parquet"
            )
            Dataset.from_dict({"text": ["aaaaaaaa", "dddddddd"]}).to_parquet(
                root / "000.parquet"
            )
            (root / "README.md").write_text("ignored\n")

            loader, files, automatic_streaming = resolve_local_dataset(str(root))
            self.assertEqual(loader, "parquet")
            self.assertEqual(
                files,
                [str(root / "000.parquet"), str(root / "001.parquet")],
            )
            self.assertTrue(automatic_streaming)

            first_args = streaming_args(root)
            second_args = streaming_args(root)
            first = load_training_dataset(first_args, TinyTokenizer())
            second = load_training_dataset(second_args, TinyTokenizer())
            self.assertIsInstance(first, IterableDataset)
            self.assertTrue(first_args.streaming)
            self.assertEqual(training_dataloader_num_workers(first, 2), 0)

            first_blocks = list(islice(first, 8))
            second_blocks = list(islice(second, 8))
            self.assertEqual(first_blocks, second_blocks)
            self.assertEqual(len(first_blocks), 8)
            self.assertTrue(
                all(len(block["input_ids"]) == 4 for block in first_blocks)
            )

            map_dataset = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]})
            self.assertEqual(training_dataloader_num_workers(map_dataset, 2), 2)

    def test_unsupported_local_file_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "training.csv"
            path.write_text("text\nhello\n")
            with self.assertRaisesRegex(ValueError, "unsupported training data suffix"):
                resolve_local_dataset(str(path))


if __name__ == "__main__":
    unittest.main()
