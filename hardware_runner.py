#!/usr/bin/env python3
"""
End-to-end helper for running the federated learning workflow on hardware.

The script combines IP assignment, client orchestration, and simple progress
reporting that the global node can observe after each training round.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, Tuple

from assign_client_ips import (
    CLIENT_SPACES,
    parse_client_ip_arguments,
    prompt_for_value,
    write_connection_file,
)
from Model import (
    get_weighted_average_of_model_and_set_weight_of_server_model_and_save_server_weights,
    initialize_global_model_01,
)
from PythonUtils import (
    clients_to_server,
    load_client_status,
    server_to_clients,
    write_client_status,
)
from run import get_the_weightage_of_each_client

CLIENT_SCRIPTS: Dict[str, str] = {
    "client1": "Client1.py",
    "client2": "Client2.py",
    "client3": "Client3.py",
    "client4": "Client4.py",
    "client5": "Client5.py",
}

CLIENT_SPACE_LOOKUP: Dict[str, str] = dict(CLIENT_SPACES)


def parse_client_ips(values, parser) -> Dict[str, str]:
    try:
        return parse_client_ip_arguments(values)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))


def collect_connection_info(args, parser) -> Tuple[str, Dict[str, str]]:
    server_ip = args.server_ip
    client_ips = parse_client_ips(args.client_ip, parser)

    if not server_ip:
        if args.non_interactive:
            parser.error("Missing --server-ip in non-interactive mode.")
        server_ip = prompt_for_value("Server IP address")

    for client_id, _ in CLIENT_SPACES:
        if client_id not in client_ips:
            if args.non_interactive:
                parser.error(f"Missing IP address for {client_id} in non-interactive mode.")
            client_ips[client_id] = prompt_for_value(f"{client_id} IP address")

    return server_ip, client_ips


def ensure_connection_files(server_ip: str, client_ips: Dict[str, str]) -> None:
    for client_id, space_name in CLIENT_SPACES:
        ip_address = client_ips[client_id]
        path = write_connection_file(space_name, server_ip, ip_address)
        print(f"[setup] {client_id} -> {ip_address} ({path})")


def run_client_process(client_id: str, weight_filename: str, server_ip: str, client_ip: str, epochs: int) -> None:
    script_name = CLIENT_SCRIPTS[client_id]
    env = os.environ.copy()
    env["CLIENT_EPOCHS"] = str(epochs)
    env["SERVER_IP"] = server_ip
    env["CLIENT_IP"] = client_ip
    env["CLIENT_ID"] = client_id

    print(f"[client] Starting {client_id} with weights '{weight_filename}' on {client_ip}")
    process = subprocess.run(
        [sys.executable, script_name, weight_filename],
        env=env,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"{client_id} exited with code {process.returncode}")


def build_status_summary(client_id: str) -> str:
    space_name = CLIENT_SPACE_LOOKUP[client_id]
    status = load_client_status(space_name)
    if not status:
        return f"{client_id}: no status reported yet."
    stage = status.get("stage", "unknown")
    epoch = status.get("epoch")
    total_epochs = status.get("total_epochs")
    metrics = status.get("metrics", {})
    summary = f"{client_id}: stage={stage}"
    if epoch and total_epochs:
        summary += f", epoch {epoch}/{total_epochs}"
    loss_value = metrics.get("loss")
    if isinstance(loss_value, (int, float)):
        summary += f", loss={loss_value:.4f}"
    val_loss_value = metrics.get("val_loss")
    if isinstance(val_loss_value, (int, float)):
        summary += f", val_loss={val_loss_value:.4f}"
    return summary


def display_client_statuses():
    print("[status] Latest client updates:")
    for client_id, _ in CLIENT_SPACES:
        print(f"  - {build_status_summary(client_id)}")


def mark_waiting_states(total_epochs: int):
    for client_id, space_name in CLIENT_SPACES:
        write_client_status(
            space_name,
            {
                "client_id": client_id,
                "stage": "waiting",
                "epoch": 0,
                "total_epochs": total_epochs,
            },
        )


def run_round(weight_filename: str, server_ip: str, client_ips: Dict[str, str], epochs: int) -> None:
    for client_id in CLIENT_SCRIPTS:
        run_client_process(client_id, weight_filename, server_ip, client_ips[client_id], epochs)
        print(f"  -> {build_status_summary(client_id)}")
    display_client_statuses()


def main():
    parser = argparse.ArgumentParser(
        description="Assign IP addresses and orchestrate federated training rounds."
    )
    parser.add_argument("--server-ip", help="IP address of the central server/aggregator.")
    parser.add_argument(
        "--client-ip",
        action="append",
        help="Client assignment in the form client1=192.168.0.21. Repeat for each client.",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds to execute.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to run on each client per round.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail instead of prompting when information is missing.",
    )

    args = parser.parse_args()
    server_ip, client_ips = collect_connection_info(args, parser)

    ensure_connection_files(server_ip, client_ips)
    mark_waiting_states(args.epochs)

    print("[server] Initialising global model.")
    initialize_global_model_01('server_model.h5')
    server_to_clients('server_model.h5')
    run_round('server_model.h5', server_ip, client_ips, args.epochs)
    weights = get_the_weightage_of_each_client(70, 100)
    print(f"[server] Client weights: {weights}")

    for round_idx in range(1, args.rounds + 1):
        print(f"[round] Aggregation step {round_idx}/{args.rounds}")
        clients_to_server()
        get_weighted_average_of_model_and_set_weight_of_server_model_and_save_server_weights(*weights)
        server_to_clients('server_weights.h5')
        run_round('server_weights.h5', server_ip, client_ips, args.epochs)

    print("[complete] Federated training finished.")


if __name__ == "__main__":
    main()
