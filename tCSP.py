import numpy as np
''' 
G is the matrix given as output after the Morlet wavlet
Dim(G) = (2M, J, T)
M = 4, m = 2M
J = num of frequency points
T = number of samples measured in trial
'''
# # Example: Create a 3D array of shape a x b x c


def Rconstitute(Gs):
    I = len(Gs)
    Hs = np.array(I,dtype = np.ndarray)
    for i, G_i in enumerate(Gs):   
        Hs[i] = G_i.transpose((1,0,2)).transpose(0,2,1) 
    return Hs

def tCSP(Hs, J):
    I = len(Hs)
    ro = np.array((I,J),dtype = np.ndarray)
    templates = np.array(J,dtype = np.ndarray)
    for j in enumerate(J):
        K_j = np.array(I, dtype = np.ndarray)
        for i, H_i in enumerate(Hs):
            K_j[i] = H_i[j]
            templates[j] = np.mean(K_j,axis=0)
            ro[i][j] = np.corrcoef(templates[j],K_j[i]) 
    return ro
    


