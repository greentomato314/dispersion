import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel, Matern,ExpSineSquared

def threshold_otsu(im, min_value=0, max_value=255):
    # ヒストグラムの算出
    gray = np.array(im)
    hist = [0]*(256)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            hist[gray[i][j]] += 1

    s_max = (0,-10)

    for th in range(256):
        
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のときの閾値を取得
    t = s_max[0]
    
    # 算出した閾値で二値化処理
    gray2 = np.copy(gray)
    gray2[gray2 < t] = min_value
    gray2[gray2 >= t] = max_value

    return gray2,t

def to_ave(lsls):
    ls = []
    for l in lsls:
        ls.append(sum(l)/len(l))
    return ls

# 画像読み込み正方形画像を想定
path = 'SEM'
names = glob.glob(path+'/*')
dispV = []

for name in names:
    imarr = np.array(Image.open(name).convert('L').resize((300,300)))
    # 二値化
    im,thres = threshold_otsu(imarr)
    # 一回当たりの計算
    def func(imgarr):
        return len(imgarr[imgarr==255])
    # 最大でとれる正方形の大きさ
    H,W = imarr.shape[0],imarr.shape[1]
    maxGrid = max(H,W)
    # 一計算あたりの回数
    v = []
    N_try=50
    X = []
    y = []
    for size in range(1,maxGrid+1):
        vals = []
        randX,randY = np.random.randint(0,H-size+1,N_try),np.random.randint(0,W-size+1,N_try)
        for i in range(N_try):
            im0 = im[randX[i]:randX[i]+size,randY[i]:randY[i]+size]
            vals.append(func(im0)/(size*size))
        varr = np.array(vals)
        y0 = np.var(varr)
        X.append(size)
        y.append(y0)
    kernel = ConstantKernel()* Matern(nu=2.5) + WhiteKernel()
    GP = GaussianProcessRegressor(kernel=kernel)
    GP.fit(np.array(X).reshape(-1, 1), np.array(y))
    G = GP.predict(np.arange(1,maxGrid+1).reshape(-1, 1))

    yls = [[] for i in range(maxGrid)]
    for i in range(maxGrid):
        yls[i].append(y[i])
    # 誤差が大きそうなところを再実験
    batch = 70
    N_re = 25

    for i in range(N_re):
        sortedy = np.argsort(np.abs(y-G))
        for j in sortedy[-batch:]:
            size = j+1
            vals = []
            randX,randY = np.random.randint(0,H-size+1,N_try),np.random.randint(0,W-size+1,N_try)
            for i in range(N_try):
                im0 = im[randX[i]:randX[i]+size,randY[i]:randY[i]+size]
                vals.append(func(im0)/(size*size))
            varr = np.array(vals)
            y0 = np.var(varr)
            yls[j].append(y0)
        y = to_ave(yls)
        kernel = ConstantKernel()* Matern(nu=2.5) + WhiteKernel()
        GP = GaussianProcessRegressor(kernel=kernel)
        GP.fit(np.array(X).reshape(-1, 1), np.array(y))
        G = GP.predict(np.arange(1,maxGrid+1).reshape(-1, 1))
    plt.plot(X,y,label=str(N_try)+name+'_exp')
    plt.plot(np.arange(1,maxGrid+1),G,label=name+'_GP')
    print(name,np.sum(np.array(X)*np.array(y)))
    dispV.append(np.sum(np.array(X)*np.array(y)))

plt.legend()
plt.show()

df = pd.DataFrame({'name':names,'disp':dispV})
df.to_csv('result2.csv',index=False)