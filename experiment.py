import os
import sys
import glob
import re
import logging
from pathlib import Path
from audio_aug.augment import get_augment_schemes

from train import main


NUMBER_RE = r"\d+"

def parse_number(chkpt_name: str, number_regex: str) -> int:
    """
    Gets the first group that matches a number in the
    checkpoint name. Simple, but should be good enough for
    this application
    """
    match = re.search(number_regex, chkpt_name).group(0)
    if not match:
        raise RuntimeError(f"Could not parse int from filename {chkpt_name}.")
    return int(match)


def get_most_recent_chkpt(chkpt_list: [str]) -> str:
    highest = -1
    for c in chkpt_list:
        filename = Path(c).name
        checkpoint = parse_number(filename, NUMBER_RE)
        if checkpoint > highest:
            highest = checkpoint
    return f"G_{highest}.pth"


def cleanup(run_name: str, log_dir: str = "./models/model"):
    """"
    Deletes all checkpoints but the last one - needed to save
    space
    """
    logging.info(f"cleaning up {run_name} checkpoints...")
    clean_dir = Path(log_dir) / run_name
    chkpt_files = glob.glob(f"{clean_dir}/epoch_*.th")
    to_keep = Path(f"{clean_dir}/{get_most_recent_chkpt(chkpt_files)}").name
    removed_count = 0
    for f in chkpt_files:
        f = Path(f).resolve()
        fname = f.name
        if fname != to_keep:
            removed_count += 1
            os.remove(f)
    logging.info(f"Removed {removed_count} checkpoints and kept {to_keep}")


if __name__ == '__main__':
    experiment_name = "e2eASR_ExpKNN"
    base_args = sys.argv[1:]  # Remove script name
    template = "audio_aug/specaugment_scheme.yml"
    template = "audio_aug/adsmote_scheme.yml"
    runs = get_augment_schemes(gammas=[0.125, 0.25, 0.5, 0.75, 1],
                               knns=[10],
                               num_runs=1,
                               template_file=template,
                               name_prefix=experiment_name)

    for run_num, augmentor in enumerate(runs):
        # run_args = base_args.copy()
        # run_args.append('--name')
        # run_args.append(augmentor.config['run_name'])
        main(augmentor)
        cleanup(augmentor.config['run_name'])

