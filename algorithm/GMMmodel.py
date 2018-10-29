import matplotlib
import matplotlib.pyplot as plt
import random
from itertools import count
from scipy.stats import norm
from scipy.stats import multivariate_normal

is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
    from IPython import display
plt.ion()

class GMMmodel(object):
    """
    n-dimension biomodal GMM model fit with MSE.
    """
    def __init__(self, mean_0=None, mean_1=None, var_0=None, var_1=None, pi=None):    
        """
        intialize model params with parameters provided
        """            
        self._mean_0 = mean_0
        self._mean_1 = mean_1
        self._var_0 = var_0
        self._var_1 = var_1
        self._pi = pi 
        
        self._data = None    
        self._n_data = None    
        self._step = None
        self._gamma = None    # cluster responsibility (E P(zi=1 | xi))
        
        self._mean_0_recall = []
        self._mean_1_recall = []
        self._log_lik_recall = []
        self.multi_norm = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
    
    def step(self, data):        
        """
        run E-step and M-step alternatively for some number of iterations
        and terminate accordinly 
        Args:
            data[nparray]: data to fit, shape (n_data, n_feature)
        Return: 
            mean_0, mean_1, var_0, var_1, pi
        """
        self._n_data, _n_feature = data.shape
        self._data = data  
        
        co_variance = np.cov(data, rowvar=True) # shpae=(n_feature, n_feature)
        
        if self._mean_0 is None:
            mean_x, mean_y = np.mean(data[:,0]), np.mean(data[:,1])
            std_x, std_y = np.std(data[:,0]), np.std(data[:,1])
            self._pi = np.random.uniform(0, 1, size=(1))                       # pi shape=(1, )
            self._mean_0 = np.array([np.random.normal(mean_x, std_x),np.random.normal(mean_y, std_y)])   
            self._mean_1 = np.array([np.random.normal(mean_x, std_x),np.random.normal(mean_y, std_y)])   # mean shape=(n_feature, )
            cov = np.cov(data, rowvar=False)
#             self._var_0 =  np.random.normal(10, 1, size=(1))           # var shape=(1,)
#             self._var_1 = np.random.normal(10, 1, size=(1))   
            self._var_0 =  np.random.normal([10,10], 1, size=(2))           # var shape=(1,)
            self._var_1 = np.random.normal([10,10], 1, size=(2))   
        
        # write initial mean and log value
        self._mean_0_recall.append(self._mean_0.tolist())
        self._mean_1_recall.append(self._mean_1.tolist())
        l = self.log_likelihood()
        self._log_lik_recall.append(l)
        
        MAX_ITER = 50
        for t in count():  
            print("Iteration {}".format(t))
            # EM step
            self.estimate()
            self.update()
            
            # Compute new log likelihood
            l_next = self.log_likelihood()
            print ("log-likelihood:    {}".format(l_next))
            self._log_lik_recall.append(l_next)
            self._mean_0_recall.append(self._mean_0.tolist())
            self._mean_1_recall.append(self._mean_1.tolist())
            
            self.plot()
            plt.pause(0.001)  
            if is_ipython:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            
            # Terminate if not noticeable change
            if (l_next - l) < 1e-6 or t > MAX_ITER: 
                self._step = t
                break 
            l  = l_next
    
    def estimate(self):   
        """E-step, compute posterior dist of z given model params theta."""        
        z_1 = (self._data - self._mean_1)/self._var_1**0.5       # shape=(n_samples, n_features)
        z_0 = (self._data - self._mean_0)/self._var_0**0.5
        a = self._pi * self.multi_norm.pdf(z_1)
        b = a + (1-self._pi) * self.multi_norm.pdf(z_0) 
        
        self._gamma = a / b    # shape=(n_sample, )
                
    def update(self):
        """M-step, pick theta by MLE given posterior over z.
           update model params mean_0, mean_1, var_0, var_1"""
        wi1 = self._gamma.reshape((-1,1))                    #  shape (n_sample, 1)
        wi0 = (1 - self._gamma).reshape((-1,1))              #  shape (n_sample, 1)
        self._mean_1 =  np.sum(wi1 * self._data, 0) / wi1.sum(axis=0)     # shape (n_feature, )    
        self._mean_0 =  np.sum(wi0 * self._data, 0) / wi0.sum(axis=0)           
        self._var_1 =  np.sum(wi1 * np.sum((self._data - self._mean_1)**2, 1).reshape((-1,1)), 0) / wi1.sum(axis=0) # shape (1,)
        self._var_0 =  np.sum(wi0 * np.sum((self._data - self._mean_0)**2, 1).reshape((-1,1)), 0) / wi0.sum(axis=0)
        
        self._pi = np.sum(wi1, 0) / self._n_data  # shape=(1,)           
        print ("mean1: {},    mean0: {}\nvar1: {},         var0: {}".format(self._mean_1, self._mean_0, self._var_1, self._var_0))
    
    def log_likelihood(self):
        z_1 = (self._data - self._mean_1) / self._var_1**0.5   
        z_0 = (self._data - self._mean_0) / self._var_0**0.5
        
        p_1 = self._pi * self.multi_norm.pdf(z_1)
        p_0 = (1-self._pi) * self.multi_norm.pdf(z_0)       
        l = np.log(p_0+p_1).sum()   
        return l
        
    def plot(self):
        fig = plt.figure(figsize=(12,12))
        plt.clf()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        
        x1,y1 = np.array(self._mean_0_recall).T
        x2,y2 = np.array(self._mean_1_recall).T
        ax1.plot(x1,y1,'-o')
        ax1.plot(x2,y2,'-o')
        for x,y in [[x1,y1],[x2,y2]]:
            for i in range(len(x)):
                ax1.annotate(str(i),(x[i],y[i]))
        ax1.legend(['mean_0', 'mean_1'], loc='upper left')
        ax2.plot(self._log_lik_recall,'-o')
        ax2.set_ylim(bottom=-1200, top=-600)
        ax1.set_title('Trajectory of model parameter \nover iterations during EM')
        ax1.set_ylabel('means')
        ax2.set_ylabel('Log-Likelihood')                
        plt.show()
