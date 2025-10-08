#!/usr/bin/env python3
"""
Utility script to record the IP address for each client and the coordinating server.

The script writes a small JSON file named ``connection.json`` into every
``data/client*_space`` directory so each Raspberry Pi (or other device) can read
its configuration without changing the existing training code.

Example (non-interactive) usage:

    python assign_client_ips.py \
        --server-ip 192.168.0.10 \
        --client-ip client1=192.168.0.21 \
        --client-ip client2=192.168.0.22 \
        --client-ip client3=192.168.0.23 \
        --client-ip client4=192.168.0.24 \
        --client-ip client5=192.168.0.25

Run the script without arguments to be prompted for any missing values.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Tuple

from PythonUtils import space_path

CLIENT_SPACES: Tuple[Tuple[str, str], ...] = (
    ("client1", "client1_space"),
    ("client2", "client2_space"),
    ("client3", "client3_space"),
    ("client4", "client4_space"),
    ("client5", "client5_space"),
)

CONFIG_FILENAME = "connection.json"


def parse_client_ip_arguments(arguments: Iterable[str]) -> Dict[str, str]:
    """Parse ``client-id=ip`` pairs provided on the command line."""
    mapping: Dict[str, str] = {}
    if not arguments:
        return mapping

    valid_client_ids = {client_id for client_id, _ in CLIENT_SPACES}
    for raw_value in arguments:
        if "=" not in raw_value:
            raise argparse.ArgumentTypeError(
                f"Expected client mapping in the form 'client1=192.168.0.2', got '{raw_value}'."
            )
        client_id, ip_address = raw_value.split("=", 1)
        client_id = client_id.strip()
        ip_address = ip_address.strip()
        if client_id not in valid_client_ids:
            raise argparse.ArgumentTypeError(
                f"Unknown client identifier '{client_id}'. Expected one of: {sorted(valid_client_ids)}"
            )
        if not ip_address:
            raise argparse.ArgumentTypeError(f"Missing IP address for '{client_id}'.")
        mapping[client_id] = ip_address
    return mapping


def prompt_for_value(label: str, default: str | None = None) -> str:
    """Ask the user for a value, using the default if one was supplied."""
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "

    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default:
            return default
        print("A value is required. Please try again.")


def write_connection_file(space_name: str, server_ip: str, client_ip: str) -> str:
    """Persist the server/client IP information into the given client space."""
    folder_path = space_path(space_name)
    config_path = os.path.join(folder_path, CONFIG_FILENAME)
    payload = {
        "server_ip": server_ip,
        "client_ip": client_ip,
    }
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2)
        config_file.write("\n")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-client connection.json files containing IP addresses."
    )
    parser.add_argument(
        "--server-ip",
        dest="server_ip",
        help="IP address of the federated learning server/aggregator.",
    )
    parser.add_argument(
        "--client-ip",
        dest="client_ips",
        action="append",
        help="Client mapping in the form client1=192.168.0.21. "
        "Repeat this option for additional clients.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail if any IP address is missing instead of prompting.",
    )

    args = parser.parse_args()
    client_ips = parse_client_ip_arguments(args.client_ips)

    server_ip = args.server_ip
    if not server_ip:
        if args.non-interactive:
            parser.error("Missing --server-ip in non-interactive mode.")
        server_ip = prompt_for_value("Server IP address")

    for client_id, space_name in CLIENT_SPACES:
        assigned_ip = client_ips.get(client_id)
        if not assigned_ip:
            if args.non-interactive:
                parser.error(f"Missing IP address for {client_id} in non-interactive mode.")
            assigned_ip = prompt_for_value(f"{client_id} IP address")

        config_path = write_connection_file(space_name, server_ip, assigned_ip)
        print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
