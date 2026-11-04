import csv
import sys
from pathlib import Path


class CSVLogger:

    def __init__(
        self,
        log_dir,
        experiment_name,
    ):

        log_dir = Path(log_dir)

        log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path = (
            log_dir
            / f"{experiment_name}.csv"
        )

        self.header_written = (
            self.path.exists()
        )

    def log(
        self,
        epoch,
        train_metrics,
        val_metrics,
    ):

        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value
                in train_metrics.items()
            },
            **{
                f"val_{key}": value
                for key, value
                in val_metrics.items()
            },
        }

        with open(
            self.path,
            "a",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=row.keys(),
            )

            if not self.header_written:

                writer.writeheader()

                self.header_written = True

            writer.writerow(row)


class TeeLogger:
    """
    Writes standard output to both terminal console and log file concurrently.
    """

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()
        if sys.stdout is self:
            sys.stdout = self.terminal