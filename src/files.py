from pathlib import Path
import os.path

class Files:
    """This class is a utility class for working with files.

    Methods
    -------
    exists(file_path)
        :return whether the specified file_path exists on the filesystem.
    is_file(file_path)
        :return whether there is a file located on the filesystem at the specified file_path.
    is_dir(file_path)
        :return whether there is a directory located on the filesystem at the specified file_path.
    """

    @staticmethod
    def exists(file_path):
        return os.path.exists(file_path)

    @staticmethod
    def is_file(file_path):
        path = Path(file_path)
        return path.is_file()

    @staticmethod
    def is_dir(file_path):
        path = Path(file_path)
        return path.is_dir()

