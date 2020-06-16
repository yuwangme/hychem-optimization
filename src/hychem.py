class CONDITION:
    ''' class for experimental conditions '''

    def __init__(self, T=None, P=None, MF={None: None}):
        self._T = T
        self._P = P
        self._MF = MF
        return

    def T(self):
        return self._T

    def P(self):
        return self._P

    def MF(self):
        return self._MF

    def __repr__(self):
        out = f"Temperature: {self._T} K\nPressure: {self._P} atm\n"
        for k, v in self._MF.items():
            out += f"{k}: {v} (fraction)\n"
        return out


class HYCHEM:
    ''' class holding hychem parameters and generates mechanism file lines '''

    def __init__(self):
        return


class OPTIMIZER:
    ''' class for optimization of hychem parameters '''

    def __init__(self):
        return
