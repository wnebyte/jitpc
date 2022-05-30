def get(default, value=None):
    """ Utility method mirroring a ternary operator.

    :param default: a value.
    :param value: a value; default=None.
    :return: value if value is not None, otherwise default.
    """
    if value is not None:
        return value
    else:
        return default

def mk_filepath(base, name, type, dataset_url, ext):
    """Utility method for creating a filepath.

    Structure of the returned filepath:
    $base/$name_$type_$dataset$ext

    :param base: the base or directory.
    :param name: the beginning of the path.
    :param type: the middle part of the path.
    :param dataset_url: is used to derive the end of the path.
    :param ext: the file extension.
    :return: a filepath.
    """
    dataset = dataset_url.split(sep='/')[-1].split(sep='.')[0].split(sep='_')[0]
    return base.rstrip('/') + '/' + name + '_' + type + '_' + dataset + ext
