def get_model_prefix(model):
    """
    Get the prefix file associated with the model name

    Parameters
    ----------
    model : str
        The name of the model

    Returns
    -------
    prefix : str
        The prefix (e.g. for config files) associated with the model
    """
    prefix = model.lower().replace('-', '_')
    return prefix
