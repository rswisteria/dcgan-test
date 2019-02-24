import chainer
from chainer import functions as F
from chainer import links as L


class Discriminator(chainer.Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(3, 64, ksize=4, stride=2, pad=1, initialW=initializer),
            c1=L.Convolution2D(64, 128, ksize=4, stride=2, pad=1, initialW=initializer),
            bn1=L.BatchNormalization(128),
            c2=L.Convolution2D(128, 256, ksize=4, stride=2, pad=1, initialW=initializer),
            bn2=L.BatchNormalization(256),
            c3=L.Convolution2D(256, 512, ksize=4, stride=2, pad=1, initialW=initializer),
            bn3=L.BatchNormalization(512),
            l4l=L.Linear(in_size=512*4*4, out_size=2, initialW=initializer)
        )

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        return self.l4l(h)
