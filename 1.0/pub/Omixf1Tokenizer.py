
from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence


class Omixf1Tokenizer:
    """
    Exhaustive nucleotide k-mer tokenizer.

    Alphabet: A, T, U, G, C, _, N
    Maximum token length: 7
    T and U cannot coexist in the same sequence.
    Vocabulary: All valid combinations of lengths 1..7.

    Tokenization strategy:
        Greedy, non-overlapping, maximum-length chunks.
    """

    ALPHABET = ("A", "T", "U", "G", "C", "_", "N")

    def __init__(self, max_length: int = 7):
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise TypeError("max_length must be an integer")

        if not 1 <= max_length <= 7:
            raise ValueError("max_length must be between 1 and 7")

        self.max_length = max_length

        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Generate every valid combination deterministically."""

        token_id = 0

        for length in range(1, self.max_length + 1):
            for chars in product(self.ALPHABET, repeat=length):

                # T and U are mutually exclusive.
                if "T" in chars and "U" in chars:
                    continue

                token = "".join(chars)

                self._token_to_id[token] = token_id
                self._id_to_token.append(token)

                token_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def vocabulary(self) -> dict[str, int]:
        """Read-only-by-convention vocabulary mapping."""
        return self._token_to_id.copy()

    def validate(self, sequence: str) -> str:
        """Validate and normalize an input nucleotide sequence."""

        if not isinstance(sequence, str):
            raise TypeError("Sequence must be a string")

        sequence = sequence.upper()

        invalid = set(sequence) - set(self.ALPHABET)

        if invalid:
            raise ValueError(
                f"Invalid characters: {sorted(invalid)}"
            )

        if "T" in sequence and "U" in sequence:
            raise ValueError(
                "A sequence cannot contain both T and U"
            )

        return sequence

    def tokenize(self, sequence: str) -> list[str]:
        """
        Split sequence into maximum-length, non-overlapping tokens.
        """

        sequence = self.validate(sequence)

        return [
            sequence[i:i + self.max_length]
            for i in range(0, len(sequence), self.max_length)
        ]

    def encode(self, sequence: str) -> list[int]:
        """Convert a nucleotide sequence into token IDs."""

        tokens = self.tokenize(sequence)

        return [self._token_to_id[token] for token in tokens]

    def encode_batch(
        self, sequences: Iterable[str]
    ) -> list[list[int]]:
        """Encode multiple sequences."""

        return [self.encode(sequence) for sequence in sequences]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Reconstruct the original nucleotide sequence."""

        result = []

        for token_id in token_ids:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.vocab_size
            ):
                raise ValueError(
                    f"Invalid token ID: {token_id}"
                )

            result.append(self._id_to_token[token_id])

        return "".join(result)

    def convert_ids_to_tokens(
        self, token_ids: Sequence[int]
    ) -> list[str]:
        """Convert token IDs back to individual tokens."""

        return [
            self._id_to_token[token_id]
            if isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < self.vocab_size
            else self._raise_invalid_id(token_id)
            for token_id in token_ids
        ]

    @staticmethod
    def _raise_invalid_id(token_id):
        raise ValueError(f"Invalid token ID: {token_id}")

