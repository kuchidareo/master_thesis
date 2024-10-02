import cv2
from datasets import Dataset 
import numpy as np
import torchvision

text = """Lorem ipsum dolor sit amet. consectupidatat non proident, sunt in culpa qui offici."""
secret_message_delimeter = "#####"

def hide_data(image, secret_message):
    # Calculate the max bytes to be encoded.
    n_bytes = image.shape[0] * image.shape[1] // 2
    # Check if the number of bytes to encode is less than th emax bytes in the image.
    if len(secret_message) > n_bytes:
        raise ValueError("Error encountered insufficient bytes, need bigger image or less data.")
    secret_message += secret_message_delimeter
    data_index = 0
    # Convert input data to binary format using messageToBinary().
    binary_secret_msg = message_to_binary(secret_message)
    data_len = len(binary_secret_msg)
    for values in image:
        for pixel in values:
            r, g, b = message_to_binary(pixel)
            # Modify the least significant bit only if there is still data to store.
            if data_index < data_len:
                # Hide the data into red pixel.
                pixel[0] = int(binary_secret_msg[data_index] + r[1:], 2)
                data_index += 1
            if data_index < data_len:
                pixel[1] = int(binary_secret_msg[data_index] + g[1:], 2)
                data_index += 1
            if data_index < data_len:
                pixel[2] = int(binary_secret_msg[data_index] + b[1:], 2)
                data_index += 1
            if data_index >= data_len:
                break
    return image

def message_to_binary(message):
    if type(message) == str:
        return ''.join([format(ord(i), "08b") for i in message])
    elif type(message) == bytes or type(message) == np.ndarray:
        return [format(i, "08b") for i in message]
    elif type(message) == int or type(message) == np.uint8:
        return format(message, "08b")
    else:
        raise TypeError("Input type not supported.")

def show_data(image):
    binary_data = ""
    for values in image:
        for pixel in values:
            r, g, b = message_to_binary(pixel)
            binary_data += r[0] # Extracting data from the least significant bit of red pixel.
            binary_data += g[0]
            binary_data += b[0]

    # Split by 8bits
    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]
    # Convert from bits to characters.
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == secret_message_delimeter:
            break
    # Remove delimeter to show the original hidden message.
    return decoded_data[:-5] 


def attack(trainset, dataset_info, rate):
    X_train = trainset[dataset_info["data_key"]]
    y_train = trainset[dataset_info["label_key"]]
    trans_to_pil = torchvision.transforms.ToPILImage()
    trans_to_tensor = torchvision.transforms.PILToTensor()

    poisoned_count = int(len(X_train) * rate)
    random_index = np.random.choice(len(X_train), poisoned_count, replace=False)

    for index in random_index:
        image = trans_to_pil(X_train[index])
        steganographed_image = np.array(image).copy()
        steganographed_image = hide_data(steganographed_image, text)
        assert text == show_data(steganographed_image)
        X_train[index] = trans_to_tensor(steganographed_image)
    
    poisoned_trainset = Dataset.from_dict({
        dataset_info["data_key"]: X_train,
        dataset_info["label_key"]: y_train
    })
    poisoned_trainset.set_format(type="torch", columns=[dataset_info["data_key"], dataset_info["label_key"]])

    return poisoned_trainset