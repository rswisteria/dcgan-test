## dcgan-test

Chainerで[DCGAN](https://arxiv.org/abs/1511.06434)を実装してみた。

- Generatorは100次元のノイズから64x64のRGB画像を生成する。
- Discriminatorは64x64のRGB画像を入力に受け取り、Generatorによって生成された画像かどうか（つまり偽物かどうか）を判定する。
- Generatorは、生成した画像がDiscriminatorによって本物と判定されるように学習を行う。
- Discriminatorは、Generatorによって生成された画像が偽物と判定されるように学習を行う。

### 参考にしたサイト等

- [はじめてのGAN](https://elix-tech.github.io/ja/2017/02/06/gan.html)
- [できるだけ丁寧にGANとDCGANを理解する](http://mizti.hatenablog.com/entry/2016/12/10/224426)
- [ChainerでDCGANを実装しMNIST風の画像を生成する](http://blog.rystylee.com/python/chainer-dcgan-mnist)
- [Chainerで顔イラストの自動生成](https://qiita.com/mattya/items/e5bfe5e04b9d2f0bbd47)
- [Chainerを使ってコンピュータにイラストを描かせる](https://qiita.com/rezoolab/items/5cc96b6d31153e0c86bc)
