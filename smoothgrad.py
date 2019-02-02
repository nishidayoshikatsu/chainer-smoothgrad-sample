import chainer
import chainer.functions as F
from chainer.variable import Variable
from chainer.links import VGG16Layers
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

chainer.config.train=False
chainer.config.enable_backprop=True

model = VGG16Layers()

image = Image.open("./img/dog_cat.png")
image = image.resize((224,224))

sampleSize = 100
noiseLevel = 0.2 # 20%
sigma = noiseLevel*255.0
print("やるぞ")

gradList = []
for _ in range(sampleSize):
    print("槍中")
    x = np.asarray(image, dtype=np.float32)
    # RGB to BGR
    x = x[:,:,::-1]
    # 平均を引く
    x -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    x = x.transpose((2, 0, 1))
    x = x[np.newaxis]
    # ノイズを追加
    x += sigma*np.random.randn(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
    x = Variable(np.asarray(x))
    # FPして最終層を取り出す
    y = model(x, layers=['prob'])['prob']
    # 予測が最大のラベルでBP
    t = np.zeros((x.data.shape[0]),dtype=np.int32)
    #t[:] = np.argmax(y.data)
    y_data = y.data
    #print(y_data)
    #print(y_data.shape)
    t[:] = y_data.argsort()[0][-2]      # 2番目に高いラベル
    t = Variable(np.asarray(t))
    loss = F.softmax_cross_entropy(y,t)
    loss.backward()
    # 勾配をリストに追加
    grad = np.copy(x.grad)
    gradList.append(grad)
    # 勾配をクリア
    model.cleargrads()

print("完了")
G = np.array(gradList)
M = np.mean(np.max(np.abs(G),axis=2),axis=0)
M = np.squeeze(M)
plt.imshow(M,"gray")
#plt.savefig("sdog.png")
plt.savefig("scat.png")