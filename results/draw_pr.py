import _pickle as cPickle
import matplotlib.pyplot as plt

fr = open('annots.pkl', 'rb')  # 这里open中第一个参数需要修改成自己生产的pkl文件
inf = cPickle.load(fr)
fr.close()

x = inf['rec']
y = inf['prec']
plt.figure()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR cruve')
plt.plot(x, y)
plt.show()

print('AP：', inf['ap'])