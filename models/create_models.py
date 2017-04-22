def create_model(args, label):
    model = None
    if label == 'g_image':
        from .models import _netGImage
        model = _netGImage(args)
    if label == 'g_latent':
        from .models import _netGLatent
        model = _netGLatent(args)
    if label == 'd_image':
        from .models import _netDImage
        model = _netDImage(args)
    if label == 'd_latent':
        from .models import _netDLatent
        model = _netDLatent(args)
    model.initialize(args)
    print("model [%s] was created" % (label))
    return model
