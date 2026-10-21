import logging
from collections.abc import Mapping
from typing import Any

from lightning.pytorch.utilities import rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """Logger that prefixes messages with the process rank.

    With `rank_zero_only=True` (default) messages are only emitted on rank 0, so multi-GPU runs
    don't print everything once per process.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = True,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(logger=logging.getLogger(name), extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        if not self.isEnabledFor(level):
            return
        rank = getattr(rank_zero_only, "rank", 0)
        if self.rank_zero_only and rank != 0:
            return
        msg, processed = self.process(msg, kwargs)
        self.logger.log(level, f"[rank: {rank}] {msg}", *args, **processed)
