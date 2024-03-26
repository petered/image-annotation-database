import os


def asset_path_for_testing(relative: str, base_folder_relative_to_this_file: str = os.path.abspath(__file__ + '/../../../../assets')):
    """ Note - This function assumes the existance of an asset folder """

    path = os.path.abspath(os.path.join(
        base_folder_relative_to_this_file,
        # os.environ.get("_MEIPASS", default_base_folder),
        os.path.join(relative)
    ))
    return path
