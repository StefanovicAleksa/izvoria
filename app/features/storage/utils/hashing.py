import hashlib

CHUNK_SIZE = 65536 #64KB

def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hashing algorihthm: {algorithm}")

    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                hasher.update(chunk)

        return hasher.hexdigest()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied reading file: {file_path}")