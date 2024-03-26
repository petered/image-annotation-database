import hashlib
import os
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterator, Sequence, Union, Optional, Mapping, Callable, Tuple, Hashable

import cv2
import ulid
from more_itertools import first

from artemis.general.custom_types import BGRImageArray
from artemis.general.hashing import compute_fixed_hash
from artemis.general.should_be_builtins import remove_prefix
from artemis.plotting.tk_utils.machine_utils import is_windows_machine


def get_file_cached(path: str, cache_dir = os.path.expanduser("~/.eagle_eyes_cache")):
    code = compute_fixed_hash(path)
    fname, ext = os.path.splitext(os.path.basename(path))
    new_name = os.path.join(cache_dir, f"{fname}_{code}{ext}")
    new_path = os.path.join(cache_dir, new_name)
    if os.path.exists(new_path):
        return new_path
    else:
        assert os.path.exists(path), f"Cannot find file {path}, and do not have a cached copy."
        os.makedirs(cache_dir, exist_ok=True)
        shutil.copy(path, new_path)
    return new_path


def get_file_from_local_or_backup(local_file_path: str, local_folder: str, backup_folder: str, copy: bool = True):
    """
    Get a file from the local folder, or if it is not there, from the backup.

    """
    local_folder = os.path.expanduser(local_folder)
    local_file_path = os.path.expanduser(local_file_path)
    assert local_file_path.startswith(local_folder), f"Local file path {local_file_path} is not in local folder {local_folder}"
    local_folder = os.path.expanduser(local_folder)
    primary_path = os.path.expanduser(local_file_path)
    if os.path.exists(primary_path):
        return primary_path
    else:
        secondary_path = os.path.join(backup_folder, remove_prefix(primary_path, local_folder).lstrip(os.sep))
        if os.path.exists(secondary_path):
            if copy:
                print(f"Copying from {primary_path} to {secondary_path}...")
                os.makedirs(os.path.dirname(primary_path), exist_ok=True)
                if os.path.isdir(secondary_path):
                    shutil.copytree(secondary_path, primary_path)
                else:
                    shutil.copy2(secondary_path, primary_path)
                print("Done Copying.")
                return primary_path
            else:
                return secondary_path
        elif not os.path.exists(backup_folder):
            raise FileNotFoundError(f"Cannot find video {primary_path} locally and the backup folder {backup_folder} does not exist.  Is your backup drive plugged in?")
        else:
            raise FileNotFoundError(f"Cannot find video {primary_path} locally and or in the backup location {secondary_path}, though the backup folder exists.")



def sync_dirs(src: Union[str, Path], dst: Union[str, Path]) -> None:
    src = Path(src)  # Convert src to Path object if it's a string
    dst = Path(dst)  # Convert dst to Path object if it's a string

    if not dst.exists():
        dst.mkdir(parents=True)

    for item in src.iterdir():
        s = src / item
        d = dst / item.name

        print(f"Syncing {s} to {d}")

        if s.is_dir():
            sync_dirs(s, d)
        else:
            if not d.exists() or s.stat().st_mtime - d.stat().st_mtime > 1:
                shutil.copy2(s, d)  # copy2 will also copy metadata


def get_hash_for_file(file_path: str, byte_sampling_threshold: int = 100000, total_samples = 3) -> bytes:
    """
    Generates a fast, base32-encoded, content-based hash for a file.

    This will use sampling to speed up the read - at the cost of not being sensitive to all file
    contents

    Note: We used to include metadata like modification time in this hash
    But - Windows and Mac have different modification time - so we just go for contents.
    """
    assert total_samples > 1, "Total samples must be at least 2"
    hash_obj = hashlib.sha256()
    file_size = os.path.getsize(file_path)
    # modification_time = os.path.getmtime(file_path)
    # creation_time = os.path.getctime(file_path)
    # Prepare metadata string and update hash
    # metadata_str = f'{file_size}{modification_time}{creation_time}'.encode()
    # metadata_str = f'{file_size}{modification_time}'.encode()
    metadata_str = f'{file_size}'.encode()

    hash_obj.update(metadata_str)
    # print(f"Hashing {file_path} with \n"
    #       f"  size {file_size} \n"
    #       # f"  modification time: {datetime.fromtimestamp(modification_time)}\n"
    #       # f"  modification time mod1:: {modification_time % 1}\n"
    #       )

    # Define how many bytes to sample from the file
    if file_size < byte_sampling_threshold:  # If the file is small enough - just read the whole thing
        with open(file_path, 'rb') as file:
            data = file.read()
            hash_obj.update(data)
    else:  # If it is larger, sample parts of the file
        sample_size = byte_sampling_threshold//total_samples
        with open(file_path, 'rb') as file:
            for i in range(total_samples):
                # Seek to a sample position based on file size and total samples
                sample_position = (file_size-sample_size) * (i // (total_samples-1))
                file.seek(sample_position)
                data = file.read(sample_size)
                # print("  sampling at ", sample_position, " of ", file_size, " bytes gives ", compute_fixed_hash(bytes_to_base32_string(data)))
                hash_obj.update(data)

    # Generate SHA-256 hash
    result: bytes = hash_obj.digest()
    # print('  to hash: ', bytes_to_base32_string(result))
    return result


def create_ulid_with_hash_randomness(
        mtime: float,  # Modification time from system
        file_hash: bytes,  # Hash of file contents
        grouping_hash: Optional[bytes] = None  # Optional grouping bytes - we only use the first 15-bits of this
    ) -> str:
    """Creates a ULID using the file's modification time and hash for randomness.

    Normally, ULIDS have
    - 48 bits to encode a ms timestamp from the Unix epoch
    - 80 bits of randomness

    We modify this a bit
    - We reduce timestamp precision, because different OS's have different modification time precision, and can be different on the order of 10ms
    - We reserve 10 bytes for "grouping", which can be used to associate files together.
    - We use the file hash as the randomness

    So, our ULIDS have
    - 40 bits to encode a ms timestamp from the Unix epoch
    - 15 bits for grouping (e.g. identifying which device the file was recorded on)
    - 73 bits of randomness  (the first 73 bits of the file hash)

    We modify the standard ULID format to shorten the timestamp from 48 bits to 40 bits,


    """
    assert len(file_hash) >= 10, "File hash must be at least 10 bytes long"
    timestamp = int.to_bytes(int(mtime*1000), ulid.constants.TIMESTAMP_LEN, "big")

    # Start with an ampty 128-bit array
    byte_array = bytearray(ulid.constants.BYTES_LEN)
    # Fill in the first 40 bits (5 bytes) with the timestamp
    byte_array[:5] = timestamp[:5]
    # Fill in the 15 grouping bits
    byte_array[5] = grouping_hash[0] if grouping_hash is not None else 0
    byte_array[6] = grouping_hash[1] & 0b1111_1110 if grouping_hash is not None else 0

    # Fill in the randomness
    byte_array[6] &= file_hash[0] & 0b0000_0001
    byte_array[7:] = file_hash[1:10]
    final_bytes = bytes(byte_array)
    # randomness = file_hash[:ulid.constants.RANDOMNESS_LEN]
    return str(ulid.ULID.from_bytes(final_bytes))


def is_not_hidden_file(file_path: str) -> bool:
    return not os.path.basename(file_path).startswith('.')


def is_modified_within_time_range(
        file_path: str,
        time_range: Tuple[Optional[float], Optional[float]],
        timestamp_getter: Optional[Callable[[str], float]] = None
) -> bool:
    """Returns True if the file was modified within the given time range."""
    start_time, end_time = time_range
    mtime = os.path.getmtime(file_path) if timestamp_getter is None else timestamp_getter(file_path)
    return mtime is not None and (start_time if start_time is not None else -float('inf')) <= mtime <= (end_time if end_time is not None else float('inf'))


def create_ulid_for_file(file_path: str, grouping_args: Optional[Hashable] = None) -> str:
    """Creates a ULID for the given file."""
    mtime = os.path.getmtime(file_path)
    file_hash = get_hash_for_file(file_path)
    if grouping_args is not None:
        hash_obj = hashlib.sha256()
        hash_obj.update(str(grouping_args).encode())
        grouping_hash = hash_obj.digest()
    else:
        grouping_hash = None
    return create_ulid_with_hash_randomness(mtime, file_hash, grouping_hash)


def create_sync_mapping_to_flat_dir(src_dir: str, dst_dir: str, src_filter_function: Optional[Callable[[str], bool]] = None
                                    ) -> Mapping[str, str]:
    mapping = {}
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            if src_filter_function is not None and not src_filter_function(src_file_path):
                continue

            # mtime = os.path.getmtime(src_file_path)
            # file_hash = get_hash_for_file(src_file_path)
            ulid_part = create_ulid_for_file(src_file_path)
            dest_file_nickname = strip_out_ulid_filename_prefix_if_any(file)
            dst_file_full_name = f"{ulid_part}_{dest_file_nickname}"
            mapping[src_file_path] = os.path.join(dst_dir, dst_file_full_name)
    return mapping


POSSIBLE_ULID_CHARS = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'


def get_potential_ulid_from_file_path(file_path: str) -> Optional[str]:
    """Returns the potential ULID from the given file path."""
    file_name = os.path.basename(file_path)
    potential_ulid = file_name.split('_')[0]
    if len(potential_ulid) == 26 and all(c in POSSIBLE_ULID_CHARS for c in potential_ulid):
        return potential_ulid
    else:
        return None


def strip_out_ulid_filename_prefix_if_any(file_path: str) -> str:
    """Strips out the ULID filename prefix from the given file path."""
    parent_dir, file_name = os.path.split(file_path)
    if get_potential_ulid_from_file_path(file_path) is not None:
        return os.path.join(parent_dir, '_'.join(file_name.split('_')[1:]))
    else:
        return file_path


def get_timestamp_or_none_from_ulid_prefixed_path(ulid_path: str) -> Optional[float]:
    """Extracts the timestamp from a ULID-prefixed path, if the path is indeed prefixed with a ULID."""
    potential_ulid = get_potential_ulid_from_file_path(ulid_path)
    if potential_ulid is None:
        return None
    return ulid.ULID.from_str(potential_ulid).timestamp


@dataclass
class BackedUpDirectory:
    local_directory: str
    backup_directory: str

    def get_file(self, relpath: str, copy: bool = True) -> str:
        return get_file_from_local_or_backup(os.path.join(self.local_directory, relpath).replace('\\', os.sep).replace('/', os.sep), self.local_directory, self.backup_directory, copy=copy)

    def get_folder(self, relpath: str, copy: bool = True) -> str:
        return get_file_from_local_or_backup(os.path.join(self.local_directory, relpath).replace('\\', os.sep).replace('/', os.sep), self.local_directory, self.backup_directory, copy=copy)

    def sync(self, path_to_sync: str, skip_if_no_secondary: bool = False) -> str:
        path_to_sync = path_to_sync.replace("\\", os.sep).replace("/", os.sep)
        primary_path = os.path.join(self.local_directory, path_to_sync)
        secondary_path = os.path.join(self.backup_directory, path_to_sync.lstrip(os.sep))
        if os.path.exists(secondary_path) or not skip_if_no_secondary:
            sync_dirs(secondary_path, primary_path)
        return primary_path

    def get_backup_path(self, relpath: str) -> str:
        return os.path.join(self.backup_directory, relpath)


def iter_images_from_video(path: str) -> Iterator[BGRImageArray]:
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def iter_images_from_image_sequence(paths: Sequence[str]) -> Sequence[str]:
    for path in paths:
        yield cv2.imread(path)


EXTERNAL_DRIVE_PATH = '/Volumes/WD_4TB' if not is_windows_machine() else 'E:\\'


@dataclass
class DroneDataDirectory(BackedUpDirectory):
    local_directory: str = os.path.join(os.path.expanduser('~'), 'drone')
    backup_directory: str = os.path.expanduser(os.path.join(EXTERNAL_DRIVE_PATH, 'drone'))


def get_associated_srt_file_or_none(video_path: str) -> Optional[str]:
    """
    srt files should match the video path with a .srt or .SRT extension


    :param video_path:
    :return:
    """
    video_path_without_ulid_prefix = strip_out_ulid_filename_prefix_if_any(video_path)

    parent_dir = os.path.dirname(video_path)
    video_name, _ = os.path.splitext(os.path.basename(video_path_without_ulid_prefix))
    # Find files in parent directory with a case-insensitive match to the video name

    potential_srt_files = list(glob(os.path.join(parent_dir, f"*{video_name}.srt"))) + list(glob(os.path.join(parent_dir, f"*{video_name}.SRT")))
    n_matches = len(potential_srt_files)
    if n_matches == 0:
        return None
    elif n_matches == 1:
        return potential_srt_files[0]
    else:
        # This is not great, but as a first pass which is only used when there are multiple matches, it's ok for now.
        print(f"Warning: {n_matches} potential SRT files found for video {video_path} ({potential_srt_files}).  \n\nChoosing the closest one by timestamp.")
        had_ulid_prefix = video_path_without_ulid_prefix != video_path
        if had_ulid_prefix:
            # If there are multiple potential SRT files
            video_mtime = get_timestamp_or_none_from_ulid_prefixed_path(video_path)
            potential_srt_mtimes = {f: get_timestamp_or_none_from_ulid_prefixed_path(f) for f in potential_srt_files}
        else:
            video_mtime = os.path.getmtime(video_path)
            potential_srt_mtimes = {f: os.path.getmtime(f) for f in potential_srt_files}

        closest_srt_file = min(potential_srt_mtimes, key=lambda f: abs(potential_srt_mtimes[f] - video_mtime))
        print(f"Chosen SRT file {closest_srt_file} had {abs(potential_srt_mtimes[closest_srt_file] - video_mtime):.1f} seconds difference from video {video_path}")
        return closest_srt_file

    # potential_srt_files = (os.path.splitext(video_path)[0] + '.srt', os.path.splitext(video_path)[0] + '.SRT')
    # srt_path = first((p for p in potential_srt_files if os.path.exists(p)), default=None)
    # return srt_path
