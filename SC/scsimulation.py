import numpy as np
import matplotlib.pyplot as plt

from SC.gnlse import gnlse

class Simulation:
    def __init__(self, wavelength, t0, power, flength, loss, C, betas, gamma):
        # Parameter Settings
        self.A = None
        self.Z = None
        self.AT = None
        self.AW = None
        self.W = None
        self.n = 2 ** 16
        self.t0 = t0
        self.C = C
        self.gamma = gamma
        self.betas = betas
        self.loss = loss
        self.flength = flength * 1e-3
        self.nsaves = 200
        self.twidth = 60
        self.c = 299792458 * 1e9 / 1e12
        self.T = np.linspace(-self.twidth / 2, self.twidth / 2, self.n)
        self.Ao = np.sqrt(power)
        self.A = self.Ao * np.exp(-((1 + 1j * self.C) / 2) * (self.T / self.t0) ** 2)
        self.w0 = (2.0 * np.pi * self.c) / wavelength
        self.fr = 0.148
        self.tau1 = 0.023
        self.tau2 = 0.1645
        self.RT = ((self.tau1 ** 2 + self.tau2 ** 2) / self.tau1 / self.tau2 ** 2 *
                   np.exp(-self.T / self.tau2) * np.sin(self.T / self.tau1))
        self.RT[self.T < 0] = 0

    def gauss(self):
        self.A = self.Ao * np.exp(-((1 + 1j * self.C) / 2) * (self.T / self.t0) ** 2)

    def sech(self):
        self.A = self.Ao * np.exp(-1j * self.C / 2 * (self.T / self.t0) ** 2) / np.cosh(self.T/self.t0)

    def gnlse_cal(self, rtol=1e-3, atol=1e-4):
        self.Z, self.AT, self.AW, self.W = gnlse(self.T, self.A, self.w0, self.gamma, self.betas, self.loss, self.fr,
                                                 self.RT, self.flength,
                                                 self.nsaves, rtol, atol).run()

    def gnlse_plot(self):
        lIW = (abs(self.AW) ** 2)
        wmin = 600
        wmax = 4500
        WL = 2 * np.pi * self.c / self.W
        iis = (WL > wmin) & (WL < wmax)
        lIT = abs(self.AT) ** 2

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)

        # Figure 1
        fig1 = ax1.pcolor(WL[iis], self.Z, lIW[:,iis], cmap='jet')
        ax1.set_xlim([wmin, wmax])
        plt.colorbar(fig1, ax=ax1)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Waveguide length (mm)')

        # Figure 2
        fig2 = ax2.pcolor(self.T, 1000*self.Z, lIT, cmap='jet')
        ax2.set_xlim([-30, 30])
        plt.colorbar(fig2, ax=ax2)
        ax2.set_xlabel('Time Delay (ps)')
        ax2.set_ylabel('Waveguide length (mm)')

        # Figure 3
        ax3.plot(self.T, lIT[self.nsaves - 1, :])
        ax3.set_xlim([-40, 40])
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Power (W)')
        pulse = np.zeros([len(self.T), 2])
        pulse[:, 0] = self.T
        pulse[:, 1] = abs(self.AT[self.nsaves - 1,:]) ** 2
        np.savetxt('pulse_17000W.txt', pulse)

        # Figure 4
        ax4.plot(WL[iis], lIW[self.nsaves - 1, iis])
        ax4.set_xlim([0, 5000])
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Intensity (a.u.)')
        Spect = np.zeros([len(WL[iis]), 2])
        Spect[:, 0] = WL[iis]
        Spect[:, 1] = lIW[self.nsaves - 1, iis]
        np.savetxt('spect_17000W.txt', Spect)


if __name__ == '__main__':
    betas = [-3.9629261e-01, 3.9372774e-03, -9.7895932e-06, 7.1608533e-08, -1.1079456e-10, -6.1391192e-12,
             1.3216948e-14, 5.1218183e-16]
    inst = Simulation(wavelength=1960, t0=0.06, power=800, flength=23.77, loss=400, C=0, betas=betas, gamma=20.7248)
    inst.gnlse_cal()
    inst.gnlse_plot()
    plt.show()
