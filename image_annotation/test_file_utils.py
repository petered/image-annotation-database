import os
import shutil

from ulid import ULID

from image_annotation.file_utils import get_hash_for_file, create_sync_mapping_to_flat_dir, is_not_hidden_file, create_ulid_for_file, DroneDataDirectory
from artemis.fileman.file_utils import iter_sync_files, flip_bit_in_file, walk_fullpath
from artemis.general.debug_utils import easy_profile
from artemis.general.hashing import compute_fixed_hash
from artemis.general.utils_for_testing import hold_tempfile, hold_tempdir
from artemis.image_processing.image_utils import imread_any_path


def test_read_file_with_non_ascii_path():
    # Latin path we intentionally mess with fore/back slashes here
    path = DroneDataDirectory().get_file('test_data\casara_on/DJI_202305261131_040_Targets/DJI_20230526120735_0001_W.JPG')
    image = imread_any_path(path)
    assert image is not None
    assert image.shape == (3000, 4000, 3)

    # Path containing cyrillic
    path = DroneDataDirectory().get_file('test_data/test_дir/dji_2023-07-06_18-09-26_0007.jpg')
    image = imread_any_path(path)
    assert image is not None
    assert image.shape == (3024, 4032, 3)


def test_get_hash_for_file():
    path = DroneDataDirectory().get_file('test_data/test_дir/dji_2023-07-06_18-09-26_0007.jpg')
    with easy_profile(f"Computing hash for {path}"):
        file_hash_code = get_hash_for_file(path)
    assert file_hash_code == b'\x99\xb9\xd4\xbb.\x8a\x81\xe4\x8e\x803\x93/\x10r\x98b+\xf6\x97h/\xca\x05l\x0e\xa2\xd6\x9d\xe6\xe0\r', f'Got wrong Hash: {file_hash_code}'

    path = DroneDataDirectory().get_file('test_data/test_дir/dji_2023-07-06_18-08-56_0006.mp4')
    with easy_profile(f"Computing hash for {path}"):
        file_hash_code = get_hash_for_file(path)
    assert file_hash_code == b"'\xf3}\x96\x86\xa0\xae^\xd3\xcdPs\xe4\xe7\xaf\r\xa3\xd7g\xfc\xa11\xbb\x91<\xedFmuD7>", f'Got wrong Hash: {file_hash_code}'


def test_create_ulid_for_file():
    path = DroneDataDirectory().get_file('test_data/test_дir/dji_2023-07-06_18-09-26_0007.jpg')
    mtime = os.path.getmtime(path)
    assert abs(mtime-1688749064.8354378) < 1, "Timestamp is off by more than a second from what we expect"
    path_ulid = create_ulid_for_file(path)
    assert path_ulid == '01H4RMKGG002WX9ESEHA0Y93M0', f'Got wrong ULID: {path_ulid}'
    assert abs(ULID.from_str(path_ulid).timestamp - mtime) < 1, "Timestamp is off more than a second"

    path_ulid_with_grouping = create_ulid_for_file(path, grouping_args=('Serial Number: 89328323', 'Camera Model: DJI FC2203'))
    assert path_ulid_with_grouping == '01H4RMKGQX02WX9ESEHA0Y93M0', f'Got wrong ULID: {path_ulid_with_grouping}'
    assert path_ulid_with_grouping[:8] == path_ulid[:8]
    assert path_ulid_with_grouping[8:10] != path_ulid[8:10]
    assert path_ulid_with_grouping[10:] == path_ulid[10:]
    assert abs(ULID.from_str(path_ulid_with_grouping).timestamp - mtime) < 1, "Timestamp is off more than a second"

    with hold_tempfile() as temp_copy_path:

        shutil.copy2(path, temp_copy_path)
        new_path_ulid = create_ulid_for_file(temp_copy_path)
        assert new_path_ulid == path_ulid, f'Got wrong ULID: {new_path_ulid}'

        # Now, lets change the last bit of the file
        flip_bit_in_file(temp_copy_path, -1)
        new_path_ulid = create_ulid_for_file(temp_copy_path)
        # ULID will of course change because it's a modification:
        assert new_path_ulid != path_ulid, f'Should have gotten a different ULID, but got {new_path_ulid}'
        # But the random bits at the end should also change
        assert new_path_ulid[-10:] != path_ulid[-10:], f'Random bits should have changed, but got {new_path_ulid[-10:]}'

        # Flip it back
        flip_bit_in_file(temp_copy_path, -1)
        new_path_ulid = create_ulid_for_file(temp_copy_path)
        # Again - different modification, different ULID
        assert new_path_ulid != path_ulid, f'Got wrong ULID: {new_path_ulid}'
        # But now random bits should be the same
        assert new_path_ulid[-10:] == path_ulid[-10:], f'Random bits should be the same, but got {new_path_ulid[-10:]}'


def test_sync_mapping():

    with hold_tempdir() as temp_dir:
        sync_mapping = create_sync_mapping_to_flat_dir(
            src_dir=os.path.join(DroneDataDirectory().local_directory, 'test_data/test_дir/'),
            dst_dir=temp_dir,
            src_filter_function=is_not_hidden_file
        )
        assert len(sync_mapping) == 16
        expected_hash = '7NCDISX55QJ6A43YB4WUUVZ7WI'
        actual_hash = compute_fixed_hash(sorted(os.path.basename(v) for v in sync_mapping.values()))
        assert actual_hash == expected_hash, f'Got wrong hash: {actual_hash}'

        # Now, lets actually sync using this mapping
        for progress in iter_sync_files(sync_mapping):
            print(progress)

        # Now, lets do another sync from the old destination to the new destination
        with hold_tempdir() as temp_dir2:
            sync_mapping2 = create_sync_mapping_to_flat_dir(
                src_dir=temp_dir,
                dst_dir=temp_dir2,
                src_filter_function=is_not_hidden_file
            )
            assert len(sync_mapping2) == 16
            # Assert that paths are the same as last time.  This verifies that:
            # - The content-based hash is deterministic
            # - We're not re-adding the ULID prefix
            assert compute_fixed_hash(sorted(os.path.basename(v) for v in sync_mapping2.values())) == expected_hash
            for progress in iter_sync_files(sync_mapping2):
                print(progress.get_sync_progress_string())


def test_walk_fullpath():

    with hold_tempdir() as fdir:
        os.makedirs(os.path.join(fdir, 'subdir1'))
        with open(os.path.join(fdir, 'subdir1', 'aaa.txt'), 'w') as f:
            f.write('aaa')
        with open(os.path.join(fdir, 'bbb.txt'), 'w') as f:
            f.write('bbb')

        assert set(walk_fullpath(fdir)) == {os.path.join(fdir, 'subdir1', 'aaa.txt'), os.path.join(fdir, 'bbb.txt')}


if __name__ == '__main__':
    test_read_file_with_non_ascii_path()
    test_get_hash_for_file()
    test_sync_mapping()
    test_create_ulid_for_file()
    test_walk_fullpath()
