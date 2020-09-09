from .CosineSimilarity import CosineSimilarity


class CosineDissimilarity(CosineSimilarity):
    def __init__(
            self,
            use_mask=True, use_scalings=False,
            batch_reduction=None, spatial_reduction=None
    ):
        super(CosineDissimilarity, self).__init__(
            use_mask=use_mask, use_scalings=use_scalings,
            batch_reduction=batch_reduction, spatial_reduction=spatial_reduction
        )
        self._set_name("cosine-dissimilarity")

    def local_deviation(self, predictions, targets, keepdim=True, eps=1.e-9):
        dev = super(CosineDissimilarity, self).local_deviation(predictions, targets, keepdim=keepdim, eps=eps)
        dev = (1 - dev) / 2
        return dev
