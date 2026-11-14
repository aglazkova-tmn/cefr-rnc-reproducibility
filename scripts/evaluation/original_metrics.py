import numpy as np

def evaluate(truth,p):
    cm=np.zeros((6,6),int);np.add.at(cm,(truth-1,p-1),1)
    support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
    divide=lambda a,b:np.divide(a,b,out=np.zeros(6,dtype=float),where=b!=0)
    precision=divide(tp,predicted);recall=divide(tp,support);f1=divide(2*tp,support+predicted)
    corrected=np.where(abs(truth-p)<=1,truth,p);rcm=np.zeros((6,6),int);np.add.at(rcm,(truth-1,corrected-1),1)
    rf1=divide(2*rcm.diagonal(),rcm.sum(0)+rcm.sum(1))
    assert rf1.mean()+1e-12>=f1.mean()
    return cm,precision,recall,f1,rf1,float(np.mean(abs(truth-p)))
