"""
some intro
"""
import numpy as np
from scipy.special import gammaln
from numpy.polynomial.hermite import hermgrid3d, hermgauss

pi = np.pi
ln2 = np.log(2)
sq2 = np.sqrt(2)

def logA(n):
    return -0.5 * (n*ln2 + gammaln(n+1) + 0.5*np.log(pi))

class BoseGas3d():
    '''
    some description
    '''
    def __init__(self, cutoff, Natoms, g_int, aspect_ratio=1, beta=None, step=None):
        self.cutoff = cutoff
        self.N = Natoms
        self.g_int = g_int
        self.AR = aspect_ratio
        if beta is None:
            beta = 1/cutoff
        self.set_beta(beta)

        self.n_basis = np.arange(cutoff)
        a = logA(self.n_basis)
        self.A = np.exp(np.einsum('i,j,k->ijk', a, a, a))
        self.x_emf, self.w_emf = hermgauss(2*cutoff-2)
        self.state = self.gen_initial_state()
        if step is None:
            self.step = self.find_h_best(self.state)
            print('compute initial MC step h: %.2g'%self.step)
        else:
            self.step = step
            print('set initial MC step h: %.2g'%self.step)
        self.energy = self.compute_energy(self.state)

    def set_beta(self, beta):
        self.beta = beta

    def norm(self, C):
        return np.sum(np.abs(C)**2)

    def renormalize(self, C):
        return C/np.sqrt(self.norm(C)/self.N)

    def gen_initial_state(self):
        C = np.zeros((self.cutoff, self.cutoff, self.cutoff), dtype=np.complex)
        C[0,0,0] = 1
        return self.renormalize(C)

    def E_ho(self, C):
        return np.einsum('ijk,i,j,k->', np.abs(C)**2, (self.n_basis + 1/2)/self.AR, self.n_basis + 1/2, self.n_basis + 1/2)

    def E_MF(self, C):
        CC = C*self.A
        # print(CC.shape)
        P =  np.abs(hermgrid3d(self.x_emf/sq2, self.x_emf/sq2, self.x_emf/sq2, CC))**4
        # print(P.shape)
        return 0.5*self.g_int * np.einsum('ijk,i,j,k->', P, self.w_emf, self.w_emf, self.w_emf)/(2*sq2)/np.sqrt(self.AR)

    def compute_energy(self, C):
        return self.E_ho(C) + self.E_MF(C)

    def to_space(self, C, x, y, z, broadcast=False):
        CC = C*self.A
        if len(CC.shape) > 3 and broadcast:
            CC = np.transpose(CC, axes=(1,2,3,0))
            return np.einsum('...ijk,i,j,k->...ijk', hermgrid3d(x, y, z, CC), np.exp(-x**2/2), np.exp(-y**2/2), np.exp(-z**2/2))
        else:
            return np.einsum('ijk,i,j,k->ijk', hermgrid3d(x, y, z, CC), np.exp(-x**2/2), np.exp(-y**2/2), np.exp(-z**2/2))



    def delta(self, h=1):
        r = np.random.rand(*self.n_basis.shape)
        return (1-0.99*self.n_basis/self.cutoff)*h*r

    def rand_phases(self):
        r = np.random.rand(*self.n_basis.shape)
        return np.exp(-1j*2*pi*r)

    def find_h_best(self, C, h=0.1, trials=10, num=50):
        o = np.ceil(np.log10(h))
        hs = np.logspace(o-3, o+1, num=num)
        ps = np.empty((trials, len(hs)))
        for i in range(trials):
            for j in range(len(hs)):
                q = self.renormalize(C + self.delta(hs[j])*self.rand_phases())
                ps[i,j] = np.exp(-(self.compute_energy(q)-self.compute_energy(C))*self.beta)
        ps = ps.mean(0)
        J = np.argmin(np.abs(ps-0.5))
        h_best = hs[J]
        return h_best

    def update_MC_parameters(self, acc_ratio):
        if acc_ratio < 0.3:
            self.step *= 0.9
            # print('update MC step h: %.2g'%self.step)
        elif acc_ratio > 0.7:
            self.step *= 1.1
            # print('update MC step h: %.2g'%self.step)

        # if acc_ratio < 0.3 or acc_ratio > 0.7:
        #     self.step = self.find_h_best(self.state)
            # print('update MC step h: %.2g'%self.step)

    def MC_move(self):
        c1 = self.state + self.delta(self.step)*self.rand_phases()
        c1 = self.renormalize(c1)
        E_old = self.energy
        E_new = self.compute_energy(c1)
        dE = E_new - E_old
        if dE < 0.0 or np.random.random() < np.exp(- self.beta * dE):
            self.state = c1
            self.energy = E_new
            return 1
        else:
            return 0
