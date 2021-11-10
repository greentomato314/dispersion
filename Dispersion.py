import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel, Matern

def threshold(imgarr,thres):
    return len(imgarr[imgarr>=thres])/(imgarr.shape[0]*imgarr.shape[1])

class Dispersion_Calulate:
    def __init__(self,imgarr:np.array,minGrid=None,maxGrid=None,sampling=None,n_try=300):
        '''
        白黒画像想定
        '''
        self.imgarr = np.copy(imgarr)
        self.H,self.W = self.imgarr.shape
        if minGrid==None:
            self.minGrid = 1
        else:
            self.minGrid = minGrid
        if maxGrid==None:
            self.maxGrid = min(self.imgarr.shape)
        else:
            self.maxGrid = maxGrid
        if sampling==None:
            self.sampling = [i for i in range(self.minGrid,self.maxGrid,(self.maxGrid-self.minGrid)//5)]
        else:
            self.sampling = sampling
        self.n_try = n_try
    
    def calc_all(self,func,**kwargs):
        X = self.sampling
        y = []
        for i in X:
            y.append(self.calc(func,i,**kwargs))
        return X,y
    
    def calc_bayes(self,func,addpoint=30,**kwargs):
        X = self.sampling
        y = []
        for i in X:
            y.append(self.calc(func,i,**kwargs))
        for i in range(addpoint):
            kernel = ConstantKernel()*RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
            GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
            GP.fit(np.array(X).reshape(-1, 1), np.array(y))
            G, SD = GP.predict(np.arange(self.minGrid,self.maxGrid).reshape(-1, 1), return_std=True)
            ind = np.argmax(SD)
            X.append(ind+1)
            y.append(self.calc(func,ind+1,**kwargs))
        kernel = ConstantKernel()*RBF() + WhiteKernel() + ConstantKernel() * DotProduct()
        GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
        GP.fit(np.array(X).reshape(-1, 1), np.array(y))
        G, SD = GP.predict(np.arange(self.minGrid,self.maxGrid).reshape(-1, 1), return_std=True)
        ind = np.argmax(SD)
        predictV = np.sum(G * np.arange(self.minGrid,self.maxGrid))
        return X,y,predictV

    def calc(self,func,size,**kwargs):
        vals = []
        for i in range(self.n_try):
            randX = np.random.randint(0,self.H-size)
            randY = np.random.randint(0,self.W-size)
            vals.append(func(self.imgarr[randX:randX+size,randY:randY+size],**kwargs))
        varr = np.array(vals)
        return np.var(varr)

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

if __name__=='__main__':
    imarr = np.array(Image.open('imgtest.jpg').convert('L'))
    DC = Dispersion_Calulate(imarr)
    im,thres = threshold_otsu(imarr)
    X,y,predictV = DC.calc_bayes(threshold,thres=thres)
    print(predictV)
    plt.scatter(X,y,label='imgtest')
    imarr = np.array(Image.open('imgtest2.jpg').convert('L'))
    DC = Dispersion_Calulate(imarr)
    im,thres = threshold_otsu(imarr)
    X,y,predictV = DC.calc_bayes(threshold,thres=thres)
    print(predictV)
    plt.scatter(X,y,label='imgtest2')
    imarr = np.array(Image.open('imgtest3.jpg').convert('L'))
    DC = Dispersion_Calulate(imarr)
    im,thres = threshold_otsu(imarr)
    X,y,predictV = DC.calc_bayes(threshold,thres=thres)
    print(predictV)
    plt.scatter(X,y,label='imgtest3')
    plt.legend()
    plt.show()
    '''
    imarr = np.array(Image.open('imgtest.jpg').convert('L'))
    DC = Dispersion_Calulate(imarr)
    im,thres = threshold_otsu(imarr)
    X,y = DC.calc_bayes(threshold,thres=thres,addpoint=100)
    print(X,y)
    plt.scatter(X,y,label='imgtest')
    imarr = np.array(Image.open('imgtest2.jpg').convert('L'))
    DC = Dispersion_Calulate(imarr)
    im,thres = threshold_otsu(imarr)
    X,y = DC.calc_bayes(threshold,thres=thres,addpoint=100)
    print(X,y)
    plt.scatter(X,y,label='imgtest2')
    imarr = np.array(Image.open('imgtest3.jpg').convert('L'))
    DC = Dispersion_Calulate(imarr)
    im,thres = threshold_otsu(imarr)
    X,y = DC.calc_bayes(threshold,thres=thres,addpoint=100)
    print(X,y)
    plt.scatter(X,y,label='imgtest3')
    plt.legend()
    plt.show()
    '''



        

    



