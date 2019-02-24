import chainer
from chainer import links as L
from chainer import functions as F


class Generator(chainer.Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(Generator, self).__init__(
            l0z=L.Linear(in_size=100, out_size=512*4*4, initialW=initializer),
            bn0=L.BatchNormalization(512*4*4),
            dc1=L.Deconvolution2D(512, 256, stride=2, ksize=4, pad=1, initialW=initializer),
            bn1=L.BatchNormalization(256),
            dc2=L.Deconvolution2D(256, 128, stride=2, ksize=4, pad=1, initialW=initializer),
            bn2=L.BatchNormalization(128),
            dc3=L.Deconvolution2D(128, 64, stride=2, ksize=4, pad=1, initialW=initializer),
            bn3=L.BatchNormalization(64),
            dc4=L.Deconvolution2D(64, 3, stride=2, ksize=4, pad=1, initialW=initializer)
        )

    def __call__(self, z):
        h = F.relu(self.bn0(self.l0z(z)))
        h = F.reshape(h, (z.data.shape[0], 512, 4, 4))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        return F.sigmoid(self.dc4(h))
