from ctypes import CDLL, c_char_p, c_int, c_ubyte, create_string_buffer
import ctypes
import os
from pathlib import Path
import platform
import time

import base64
import multiprocessing
from loguru import logger


class AesCipher:
    # load dll for ruler detection
    dll_name = 'AesCipher.dll' if platform.system().lower() == 'windows' else 'AesCipher.so'

    # dll_path = Path.cwd().joinpath(dll_name)
    is_capture = os.path.exists('capture_core')
    dll_path = Path.cwd().joinpath('capture_core' if is_capture else os.getcwd(), 'model_config', dll_name)
    if not dll_path.exists():
        if platform.system().lower() == 'windows':
            dll_name0 = 'AesCipher_NoDecipher.dll'
        else:
            dll_name0 = 'AesCipher_NoDecipher.so'
        # dll_path = Path.cwd().joinpath(dll_name)
        dll_path = Path.cwd().joinpath('capture_core' if is_capture else os.getcwd(), 'model_config', dll_name0)

    if not dll_path.exists():
        dll_path = Path(__file__).parent.joinpath(dll_name)

    # dll_path = Path.cwd().joinpath('capture_core', 'model_config', dll_name)
    process_name = multiprocessing.current_process().name
    logger.debug(f'{process_name}-{os.getpid()}: {dll_path}')

    MAX_LEN = 1024 * 1024 * 5  # 5GB

    cipher_text_func = None
    decipher_text_func = None
    cipher_bytes_func = None
    decipher_bytes_func = None
    decipher_model_func = None
    try:
        cipher_dll = CDLL(str(dll_path))

        cipher_text_func = cipher_dll.cipher_text
        cipher_text_func.argtypes = [c_char_p, c_int]
        cipher_text_func.restype = c_char_p

        decipher_text_func = cipher_dll.decipher_text
        decipher_text_func.argtypes = [c_char_p]
        decipher_text_func.restype = c_char_p

        cipher_bytes_func = cipher_dll.cipher_bytes
        cipher_bytes_func.argtypes = [c_char_p, ctypes.POINTER(c_int)]
        cipher_bytes_func.restype = ctypes.POINTER(c_ubyte)

        decipher_bytes_func = cipher_dll.decipher_bytes
        decipher_bytes_func.argtypes = [c_char_p, ctypes.POINTER(c_int)]
        decipher_bytes_func.restype = ctypes.POINTER(c_ubyte)

        decipher_bytes_mt_func = cipher_dll.decipher_bytes_mt
        decipher_bytes_mt_func.argtypes = [c_char_p, c_char_p, ctypes.POINTER(c_int)]

        decipher_model_func = cipher_dll.decipher_model
        decipher_model_func.argtypes = [c_char_p, c_int]
        decipher_model_func.restype = c_char_p

    except Exception as e:
        logger.error(f'Failed to load cipher dll: {dll_path}')
        logger.error(str(e))

    def __init__(self):
        '''constructor'''
        pass

    @classmethod
    def cipher_text(cls, plain_text):
        if AesCipher.cipher_text_func is None:
            logger.error("cipher is not supported")
            return plain_text

        plain_text_bytes = plain_text.encode('utf-8')
        encrypt_text = AesCipher.cipher_text_func(plain_text_bytes, len(plain_text_bytes))
        encrypt_text = str(encrypt_text, 'utf-8')
        return encrypt_text

    @classmethod
    def decipher_text(cls, encrypt_text):
        if AesCipher.decipher_text_func is None:
            logger.error("decipher is not supported")
            return encrypt_text

        plain_text = AesCipher.decipher_text_func(encrypt_text.encode('utf-8'))
        plain_text = str(plain_text, 'utf-8')
        return plain_text

    @classmethod
    def cipher_model(cls, model_path, encrypt_model_path=None):
        """
        if encrypt_model_path is not specified, output to model_path
        """
        encrypt_bytes = None
        with open(model_path, 'rb') as fp:
            bytes = fp.read()

            start = time.time()
            if str(bytes[0:3], encoding='utf-8') == '@v_':
                encrypt_bytes = bytes
            elif len(bytes) > cls.MAX_LEN * 2:
                # only encrypt MAX_LEN bytes
                # encrypt_bytes0 = AesCipher.cipher_text_func(bytes[0: cls.MAX_LEN], cls.MAX_LEN)
                # encrypt_bytes1 = AesCipher.cipher_text_func(bytes[-cls.MAX_LEN:], cls.MAX_LEN)

                encrypt_bytes0 = cls.cipher_bytes(bytes[0: cls.MAX_LEN], cls.MAX_LEN)
                encrypt_bytes1 = cls.cipher_bytes(bytes[-cls.MAX_LEN:], cls.MAX_LEN)

                print('cipher time: ', time.time() - start)
                # concatenate
                # encrypt_bytes = encrypt_bytes0[0:7].join([encrypt_bytes0, bytes[cls.MAX_LEN:]])
                encrypt_bytes = encrypt_bytes0[0:7].join(
                    [encrypt_bytes0, bytes[cls.MAX_LEN:-cls.MAX_LEN], encrypt_bytes1[7:]])
            else:
                encrypt_bytes = cls.cipher_bytes(bytes, len(bytes))

            print('cipher + join time: ', time.time() - start)

        if not encrypt_bytes:
            return

        encrypt_model_path = Path(encrypt_model_path)
        if not encrypt_model_path or encrypt_model_path.exists() and \
                encrypt_model_path.samefile(model_path):
            # first out to the temp path
            encrypt_model_path = Path(str(model_path) + '.tmp')
            # encrypt_model_path = model_path + ".tmp"

        with open(encrypt_model_path, 'wb') as fp:
            fp.write(encrypt_bytes)

        # rename .tmp to model_path
        if encrypt_model_path.exists() and encrypt_model_path.name.endswith('.tmp'):
            Path(model_path).unlink()
            encrypt_model_path.rename(model_path)

    @classmethod
    def cipher_bytes(cls, plain_data, length):
        length = c_int(length)
        encrypt_bytes = AesCipher.cipher_bytes_func(plain_data, ctypes.byref(length))
        encrypt_bytes = ctypes.string_at(encrypt_bytes, length.value)

        return encrypt_bytes

    @classmethod
    def decipher_model(cls, model_path):
        """
        return: deciphered string or decipher io.BytesIo
        """
        with open(model_path, 'rb') as fp:
            bytes = fp.read()

            # no encryption
            head = str(bytes[0:3], encoding='utf-8')
            if head != '@v_':
                return bytes

            #
            start = time.time()

            start_idx = bytes.find(bytes[0:7], 7)
            if start_idx > 0:
                end_idx = bytes.find(bytes[0:7], start_idx + 7)

                plain_bytes0 = cls.decipher_bytes_mt(bytes[:start_idx], start_idx)
                # plain_bytes0 = cls.decipher_bytes_with_base64(bytes[:start_idx], start_idx)
                print('decipher time: ', time.time() - start)

                length = len(bytes) - end_idx
                plain_bytes1 = cls.decipher_bytes_mt(bytes[end_idx:], length)
                # plain_bytes1 = cls.decipher_bytes_with_base64(bytes[end_idx:], length)

                print('decipher time: ', time.time() - start)

                # plain_bytes = b''.join([plain_bytes0, bytes[start_idx + 7:]])
                plain_bytes = b''.join([plain_bytes0, bytes[start_idx + 7:end_idx], plain_bytes1])
            else:
                plain_bytes = cls.decipher_bytes_mt(bytes, len(bytes))

            print('decipher + join time: ', time.time() - start)
            # plain_bytes = base64.decodebytes(plain_bytes)
            return plain_bytes

    @classmethod
    def decipher_bytes(cls, encrypt_bytes, length):
        length = c_int(length)

        # start = time.time()
        if AesCipher.decipher_bytes_func:
            plain_bytes = AesCipher.decipher_bytes_func(encrypt_bytes, length)
            plain_bytes = ctypes.string_at(plain_bytes, length.value)
        else:
            plain_bytes = encrypt_bytes

        return plain_bytes

    @classmethod
    def decipher_bytes_mt(cls, encrypt_bytes, length):
        '''
        multithread version for decipher_byte
        '''
        aligned_length = (length + 16 - 1) // 16 * 16 + 1
        decipher_bytes = create_string_buffer(b'0' * aligned_length)
        length = c_int(length)
        # start = time.time()
        if AesCipher.decipher_bytes_mt_func:
            AesCipher.decipher_bytes_mt_func(encrypt_bytes, decipher_bytes, length)
            plain_bytes = ctypes.string_at(decipher_bytes, length.value)
        else:
            plain_bytes = encrypt_bytes

        return plain_bytes

    @classmethod
    def decipher_bytes_with_base64(cls, encrypt_bytes, length):
        plain_bytes = AesCipher.decipher_model_func(encrypt_bytes, length)

        # remove head length
        plain_bytes = base64.decodebytes(plain_bytes)

        return plain_bytes


def cjson2json(cjson_path, json_path):
    annotations = None
    with open(cjson_path, 'r', encoding='utf-8') as fs:
        annotations = AesCipher.decipher_text(fs.read())
    if annotations:
        with open(json_path, 'w', encoding='utf-8') as fs:
            fs.write(annotations)


def compare_data(model_path, model_data):
    with open(model_path, 'rb') as fp:
        bytes = fp.read()

        if len(bytes) != len(model_data):
            print(len(bytes), len(model_data))
            return False

        if bytes != model_data:
            return False

        return True


if __name__ == '__main__':
    encrypt_text = AesCipher.cipher_text('abcdef')
    print(encrypt_text)
    decrypt_text = AesCipher.decipher_text(encrypt_text)
    print(decrypt_text)

    model_path = r'/data/lyc/QcDetection/model_config/deep_models/swin_small_20220426-9c4bae37.pth'
    encrypt_path = r'/data/lyc/QcDetection/model_config/deep_models/encrypted/deep_models/swin_small_20220426-9c4bae36.pth'

    model_path = r"F:\MyProject\UltrasonicProject\deep_models\limb1128.pt"
    encrypt_path = r"F:\MyProject\UltrasonicProject\deep_models\encrypt\test.pt"

    AesCipher.cipher_model(model_path, encrypt_path)
    model_data = AesCipher.decipher_model(encrypt_path)

    compare_data(model_path, model_data)

    exit()

    batch_processing = 0  # 选择  单个文件操作：1   多个文件操作：0

    if batch_processing:
        # 只进行单个cjson到json的转化
        cjson_path = r'E:\test\jinan_web_crl_json\annotations.cjson'
        json_path = r'E:\test\jinan_web_crl_json\1.json'
        cjson2json(cjson_path, json_path)

    else:
        # 将一个文件夹中的cjson都转换成对应的json,默认新的json生成在原文件夹中
        cjson_dir = r"E:\ZMF_data\3doctor"
        for root, dirs, files in os.walk(cjson_dir):
            for file in files:
                if file[-5:] == 'cjson':
                    cjson_path = root + '/' + file
                    json_path = root + '/' + file[:-5] + 'json'
                    cjson2json(cjson_path, json_path)
