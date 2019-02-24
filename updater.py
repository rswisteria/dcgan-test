import chainer
from chainer import Variable
from chainer import functions as F
import numpy

class DCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.softmax_cross_entropy(y_fake, Variable(chainer.cuda.cupy.ones(batchsize, dtype=numpy.int32)))
        chainer.report({'loss': loss}, gen)
        return loss

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        l1 = F.softmax_cross_entropy(y_fake, Variable(chainer.cuda.cupy.zeros(batchsize, dtype=numpy.int32)))
        l2 = F.softmax_cross_entropy(y_real, Variable(chainer.cuda.cupy.ones(batchsize, dtype=numpy.int32)))
        loss = l1 + l2
        chainer.report({'loss': loss}, dis)
        return loss


    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)
        z = Variable(xp.asarray(numpy.random.uniform(-1, 1, (batchsize, 100, 1, 1)).astype(numpy.float32)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
