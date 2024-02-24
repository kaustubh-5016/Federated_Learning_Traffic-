import os
import shutil

def server_to_clients(filename):
    c_path = os.getcwd()
    server_space = c_path + '\\data\\server_space\\'
    client1_space = c_path + '\\data\\client1_space\\'
    client2_space = c_path + '\\data\\client2_space\\'
    client3_space = c_path + '\\data\\client3_space\\'
    client4_space = c_path + '\\data\\client4_space\\'
    client5_space = c_path + '\\data\\client5_space\\'
    source_file = server_space+filename
    destination_file1 = client1_space+filename
    destination_file2 = client2_space+ filename
    destination_file3 = client3_space+ filename
    destination_file4 = client4_space+ filename
    destination_file5 = client5_space+ filename
    shutil.copy(source_file, destination_file1)
    shutil.copy(source_file, destination_file2)
    shutil.copy(source_file, destination_file3)
    shutil.copy(source_file, destination_file4)
    shutil.copy(source_file, destination_file5)

def clients_to_server():
    c_path = os.getcwd()
    server_space = c_path + '\\data\\server_space\\'
    client1_space = c_path + '\\data\\client1_space\\'
    client2_space = c_path + '\\data\\client2_space\\'
    client3_space = c_path + '\\data\\client3_space\\'
    client4_space = c_path + '\\data\\client4_space\\'
    client5_space = c_path + '\\data\\client5_space\\'
    source_file1 = client1_space+"client1_weights.h5"
    source_file2 = client2_space + "client2_weights.h5"
    source_file3 = client3_space + "client3_weights.h5"
    source_file4 = client4_space + "client4_weights.h5"
    source_file5 = client5_space + "client5_weights.h5"
    destination_file1 = server_space+"client1_weights.h5"
    destination_file2 = server_space+"client2_weights.h5"
    destination_file3 = server_space+"client3_weights.h5"
    destination_file4 = server_space+"client4_weights.h5"
    destination_file5 = server_space+"client5_weights.h5"
    shutil.move(source_file1, destination_file1)
    shutil.move(source_file2, destination_file2)
    shutil.move(source_file3, destination_file3)
    shutil.move(source_file4, destination_file4)
    shutil.move(source_file5, destination_file5)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)