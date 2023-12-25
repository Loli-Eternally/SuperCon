import math
import pyfftw
import numpy as np
import scipy.integrate as si
import tqdm


class gnlse:
    def __init__(self, T, A, w0, gamma, betas, loss, fr, RT, flength, nsaves, rtol, atol):
        self.A = A
        self.w0 = w0
        self.n = len(T)
        self.dT = T[1] - T[0]
        self.fr = fr
        self.RT = RT
        self.flength = flength
        self.nsaves = nsaves
        self.rtol = rtol
        self.atol = atol
        self.V = 2 * np.pi * np.arange(-self.n / 2, self.n / 2) / (self.n * self.dT)
        self.alpha = np.log(10 ** (loss / 10))
        self.B = 0
        for i in range(len(betas)):
            self.B = self.B + betas[i] / math.factorial(i + 2) * self.V ** (i + 2)
        L = 1j * self.B - self.alpha / 2
        if abs(w0) > np.finfo(float).eps:
            self.gamma = gamma / w0
            W = self.V + w0
        else:
            W = 1
        self.RW = self.n * np.fft.ifft(np.fft.fftshift(RT.T))
        self.L = np.fft.fftshift(L)
        self.W = np.fft.fftshift(W)

    def run(self):
        x = pyfftw.empty_aligned(self.n, dtype="complex128")
        X = pyfftw.empty_aligned(self.n, dtype="complex128")
        plan_forward = pyfftw.FFTW(x, X)
        plan_inverse = pyfftw.FFTW(X, x, direction="FFTW_BACKWARD")

        progress_bar = tqdm.tqdm(total=self.flength, unit='mm')

        def rhs(z, AW):
            progress_bar.n = round(z, 5)
            progress_bar.update(0)

            x[:] = AW * np.exp(self.L * z)
            AT = plan_forward().copy()
            IT = np.abs(AT) ** 2

            if self.RW is not None:
                X[:] = IT
                plan_inverse()
                x[:] *= self.RW
                plan_forward()
                RS = self.dT * self.fr * X
                X[:] = AT * ((1 - self.fr) * IT + RS)
                M = plan_inverse()
            else:
                X[:] = AT * IT
                M = plan_inverse()

            rv = 1j * self.gamma * self.W * M * np.exp(-self.L * z)

            return rv

        Z = np.linspace(0, self.flength, self.nsaves)
        solution = si.solve_ivp(
            rhs,
            t_span=[0, self.flength],
            y0=np.fft.ifft(self.A),
            t_eval=Z,
            rtol=self.rtol,
            atol=self.atol)
        AW = solution.y.T

        progress_bar.close()

        AT = np.zeros([len(AW[:, 0]), len(AW[0, :])], dtype="complex128")
        for i in range(len(AW[:, 0])):
            AW[i, :] = AW[i, :] * np.exp(self.L.T*Z[i])
            AT[i, :] = np.fft.fft(AW[i, :])
            AW[i, :] = np.fft.fftshift(AW[i, :]) / self.dT
        W = self.V + self.w0

        return Z, AT, AW, W





if __name__ == '__main__':
    T = np.linspace(-60 / 2, 60 / 2, 2 ** 16)
    tau1 = 0.023
    tau2 = 0.1645
    RT = ((tau1 ** 2 + tau2 ** 2) / tau1 / tau2 ** 2 *
          np.exp(-T / tau2) * np.sin(T / tau1))
    RT[T < 0] = 0
    c = 299792458 * 1e9 / 1e12
    betas = [-3.9629261e-01, 3.9372774e-03, -9.7895932e-06, 7.1608533e-08, -1.1079456e-10, -6.1391192e-12,
             1.3216948e-14, 5.1218183e-16]

    flength = 23.77*1e-3
    Z = np.linspace(0, flength, 200)
    Ao = np.sqrt(800)
    A = Ao * np.exp(-((1) / 2) * (T / 0.06) ** 2)

    inst = gnlse(T=T, A=A, w0=(2.0 * np.pi * c) / 1960, gamma=20.7248, betas=betas, loss=400, fr=0.1480, RT=RT,
                 flength=flength, nsaves=200)
    Z, AT, AW, W = inst.run()
    lIW = (abs(AW) ** 2)
    wmin = 600
    wmax = 4500
    WL = 2 * np.pi * c / W
    iis = (WL > wmin) & (WL < wmax)
    import matplotlib.pyplot as plt
    plt.plot(WL[iis], lIW[200 - 1, iis])
    plt.xlim([0, 5000])
    plt.show()
