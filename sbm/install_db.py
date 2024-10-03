#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from base64 import b64decode
import getpass
from pathlib import Path
from time import sleep
from github import Github
from rich import print
from rich.table import Table
import tomlkit
import os
import json
import numpy as np
import toml
from .main import ScanFields

CONFIG_PATH = Path.home() / ".config" / "sbm_dataset"
CONFIG_FILE_PATH = CONFIG_PATH / "sbm_dataset.toml"

repositories = []

# Convert numpy.int64 to Python int
def custom_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    raise TypeError(f"Type {type(obj)} not serializable")

def gen_jsonfile(base_path):
    """ Generate a JSON file containing the dataset information

    Args:
        base_path (str): The base path of the dataset
    """
    dataset = []
    scan_field = None
    for root, dirs, files in os.walk(base_path):
        ch = root.split('/')[-1]
        if files:
            data = {
                "channel": ch,
                "detectors": files
            }
            dataset.append(data)
        if ch == "boresight":
            scan_field = ScanFields.load_det("boresight", base_path=root)

    nside = int(scan_field.nside)
    duration = int(scan_field.duration)
    scan_strategy = scan_field.ss
    considered_spin_n = scan_field.spins_n
    considered_spin_m = scan_field.spins_m
    scaninfo = {
        "nside": nside,
        "duration": duration,
        "scan_strategy": scan_strategy,
        "considered_spin_n": considered_spin_n,
        "considered_spin_m": considered_spin_m,
    }
    with open(os.path.join(base_path, "sim_config.json"), 'w') as f:
        json.dump(scaninfo, f, indent=4, default=custom_encoder)
        json.dump(dataset, f, indent=4)


def retrieve_local_source():
    print()
    path = Path(
        input('Please enter the directory where file "sim_config.json" resides: ')
    ).absolute()

    if not (path / "sim_config.json").is_file():
        print(f'[red]Error:[/red] {path} does not seem to contain a "sim_config.json" file')
        create_file = input('Would you like to create a "sim_config.json" file? (y/n): ')
        if create_file.lower() == 'y':
            gen_jsonfile(path)
            print(f'[green]"sim_config.json" has been created at {path}.[/green]')
        else:
            return

    name = input("Now insert a descriptive name for this location: ")

    repositories.append({"name": name, "location": str(path.resolve())})

    print(
        f"""

[green]Repository "{name}" has been added successfully.[/green]

"""
    )

def run_main_loop() -> bool:
    prompt = """Choose a source for the database:

1.   [cyan]Local source[/cyan]

     A directory on your hard disk.

s.   [cyan]Save and quit[/cyan]

q.   [cyan]Discard modifications and quit[/cyan]

"""

    while True:
        print(prompt)
        choice = input("Pick your choice (1, s or q): ").strip()

        if choice == "1":
            retrieve_local_source()
        elif choice in ("s", "S"):
            print(
                """

Saving changes and quitting...
"""
            )
            return True

        elif choice in ("q", "Q"):
            print(
                """

Discarding any change and quitting...
"""
            )
            return False

        sleep(2)


def write_toml_configuration():
    file_path = CONFIG_FILE_PATH

    # Create the directory containing the file, if it does not exist.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wt") as outf:
        outf.write(tomlkit.dumps({"repositories": repositories}))

    print(
        f"""
The configuration has been saved into file
"{str(file_path)}"
"""
    )


def main():
    if run_main_loop():
        write_toml_configuration()
        if len(repositories) > 0:
            print("The following repositories have been configured successfully:")

            table = Table()
            table.add_column("Name")
            table.add_column("Location")

            for row in repositories:
                table.add_row(row["name"], row["location"])

            print(table)

        else:
            print("No repositories have been configured")

    else:
        print("Changes have been discarded")


if __name__ == "__main__":
    main()
