from models.resnet import resnet34, resnet50, resnet101

NAME_TO_CLASS = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
}

def get_arch(architecture: str):
    """
    Args:
        architecture: name of the model to instantiate
    """
    try:
        arch = NAME_TO_CLASS.get(architecture)
    except KeyError:
        print(f'{architecture} is not implemented')
    return arch