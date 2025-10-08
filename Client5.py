import os
import sys
from Model import *
from PythonUtils import *
from tensorflow.python.keras.models import load_model

CLIENT_ID = "client5"
CLIENT_SPACE_NAME = "client5_space"
DATASET_NAME = "WASHng"
DEFAULT_EPOCHS = int(os.getenv("CLIENT_EPOCHS", "1"))


def normalize_metrics(metrics):
    serializable = {}
    if not metrics:
        return serializable
    for key, value in metrics.items():
        try:
            serializable[key] = float(value)
        except (TypeError, ValueError):
            continue
    return serializable


if __name__ == '__main__':
    filename = str(sys.argv[1])
    client_space = space_path(CLIENT_SPACE_NAME)
    connection = load_connection_info(CLIENT_SPACE_NAME)
    server_ip = os.getenv("SERVER_IP", connection.get("server_ip", ""))
    client_ip = os.getenv("CLIENT_IP", connection.get("client_ip", ""))
    epochs = DEFAULT_EPOCHS if DEFAULT_EPOCHS > 0 else 1

    def report(stage, epoch=0, total_epochs=epochs, metrics=None):
        payload = {
            "client_id": CLIENT_ID,
            "stage": stage,
            "epoch": epoch,
            "total_epochs": total_epochs,
        }
        if server_ip:
            payload["server_ip"] = server_ip
        if client_ip:
            payload["client_ip"] = client_ip
        if metrics:
            payload["metrics"] = normalize_metrics(metrics)
        write_client_status(CLIENT_SPACE_NAME, payload)
        if epoch:
            print(f"[{CLIENT_ID}] {stage.capitalize()} epoch {epoch}/{total_epochs}")
        else:
            print(f"[{CLIENT_ID}] {stage.capitalize()}")

    def progress_callback(epoch_idx, total_epochs, event, logs):
        status = "running" if event == "start" else "epoch_completed"
        metrics = normalize_metrics(logs)
        if event == "start":
            print(f"[{CLIENT_ID}] Epoch {epoch_idx}/{total_epochs} running")
        else:
            loss_value = metrics.get("loss")
            if loss_value is not None:
                print(f"[{CLIENT_ID}] Epoch {epoch_idx}/{total_epochs} completed (loss={loss_value:.4f})")
            else:
                print(f"[{CLIENT_ID}] Epoch {epoch_idx}/{total_epochs} completed")
        status_payload = {
            "client_id": CLIENT_ID,
            "stage": status,
            "epoch": epoch_idx,
            "total_epochs": total_epochs,
            "metrics": metrics,
        }
        if server_ip:
            status_payload["server_ip"] = server_ip
        if client_ip:
            status_payload["client_ip"] = client_ip
        write_client_status(CLIENT_SPACE_NAME, status_payload)

    report("starting")
    df = load_data(client_space, DATASET_NAME)
    train_x, train_y, test_x, test_y = make_rnn_data(DATASET_NAME, df, .7, 70)
    val_x = test_x[:-100]
    val_y = test_y[:-100]
    test_x = test_x[-100:]
    test_y = test_y[-100:]
    model_path = os.path.join(client_space, f'{CLIENT_ID}_model.h5')
    weight_path = os.path.join(client_space, filename)
    if os.path.exists(model_path) and os.path.exists(weight_path):
        cm = load_model(model_path)
        cm.load_weights(weight_path)
        find_mean_squared_error(cm, '5', test_y, test_x)
        history = train_model_without_callback(cm, epochs, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, progress_callback=progress_callback)
        cm.save_weights(os.path.join(client_space, f'{CLIENT_ID}_weights.h5'))
        cm.save(model_path)
        update_and_save_the_learning_curve(client_space, history, 'loss', DATASET_NAME)
    else:
        m = load_model(weight_path)
        history = train_model_without_callback(m, epochs, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, progress_callback=progress_callback)
        m.save_weights(os.path.join(client_space, f'{CLIENT_ID}_weights.h5'))
        m.save(model_path)
        update_and_save_the_learning_curve(client_space, history, 'loss', DATASET_NAME)
    final_metrics = {
        "loss": float(history.history["loss"][-1]) if "loss" in history.history else None,
        "val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
    }
    final_metrics = {key: value for key, value in final_metrics.items() if value is not None}
    report("idle", epochs, epochs, final_metrics)
    delete_file(weight_path)
