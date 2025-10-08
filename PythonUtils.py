import json
import os
import shutil
import time


def space_path(*parts):
    """Build and ensure an absolute path to a data subdirectory."""
    path = os.path.join(os.getcwd(), 'data', *parts)
    os.makedirs(path, exist_ok=True)
    return path


def server_to_clients(filename):
    server_space = space_path('server_space')
    client_spaces = [
        space_path('client1_space'),
        space_path('client2_space'),
        space_path('client3_space'),
        space_path('client4_space'),
        space_path('client5_space'),
    ]
    source_file = os.path.join(server_space, filename)
    for client_space in client_spaces:
        shutil.copy(source_file, os.path.join(client_space, filename))


def clients_to_server():
    server_space = space_path('server_space')
    client_specs = [
        ('client1_space', 'client1_weights.h5'),
        ('client2_space', 'client2_weights.h5'),
        ('client3_space', 'client3_weights.h5'),
        ('client4_space', 'client4_weights.h5'),
        ('client5_space', 'client5_weights.h5'),
    ]
    for folder, weight_name in client_specs:
        src = os.path.join(space_path(folder), weight_name)
        dest = os.path.join(server_space, weight_name)
        if os.path.exists(src):
            shutil.move(src, dest)


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def client_status_path(space_name):
    """Return the path to the status.json file for the given client space."""
    return os.path.join(space_path(space_name), "status.json")


def write_client_status(space_name, status):
    """Persist a status dictionary for the client."""
    payload = {
        "space": space_name,
        "timestamp": time.time(),
        **status,
    }
    path = client_status_path(space_name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_client_status(space_name):
    """Load the most recent status for a client. Returns an empty dict if missing."""
    path = client_status_path(space_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def load_connection_info(space_name):
    """Return connection details written by assign_client_ips.py, if available."""
    path = os.path.join(space_path(space_name), "connection.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}
