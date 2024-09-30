import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, S2gate, ThermalLossChannel, LossChannel
import numpy as np
import matplotlib.pyplot as plt

def generate_cov(num_E, num_M2, TMSV_r, SMSV_r, SMSV_phi, BS_theta):
    prog = sf.Program(2 + num_E + num_M2)
    with prog.context as q:
        S2gate(TMSV_r, np.pi) | (q[0], q[1])
        if SMSV_phi == 'all0':
            for i in range(num_M2 + num_E):
                Sgate(SMSV_r, 0) | q[i + 2]
        elif SMSV_phi == 'allpi':
            for i in range(num_M2 + num_E):
                Sgate(SMSV_r, np.pi) | q[i + 2]
        elif SMSV_phi == 'odd0evenpi':
            count = 0
            while count < num_M2 + num_E:
                if count % 2 == 0:
                    Sgate(SMSV_r, 0) | q[count + 2]
                else:
                    Sgate(SMSV_r, np.pi) | q[count + 2]
                count += 1            
        else:
            raise ValueError('SMSV_phi pattern unrecognized')
        for i in range(1, num_M2 + num_E + 1):
            for ii in range(i, 0, -1):
                BSgate(BS_theta) | (q[ii], q[ii + 1])
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    return result.state.cov()

def generate_cov_with_noises(
    # Circuit parameters:
    num_E:int, num_M2:int, TMSV_r:float, SMSV_r:float, SMSV_phi:str, BS_theta:float,
    # Loss Parameters:
    T_wire:float, T_bs:float, T_end_face:float, T_detector:float, nbar:float
    )->np.array:
    prog = sf.Program(2 + num_E + num_M2)
    with prog.context as q:
        S2gate(TMSV_r, np.pi) | (q[0], q[1])
        if SMSV_phi == 'all0':
            for i in range(num_M2 + num_E):
                Sgate(SMSV_r, 0) | q[i + 2]
        elif SMSV_phi == 'allpi':
            for i in range(num_M2 + num_E):
                Sgate(SMSV_r, np.pi) | q[i + 2]
        elif SMSV_phi == 'odd0evenpi':
            count = 0
            while count < num_M2 + num_E:
                if count % 2 == 0:
                    Sgate(SMSV_r, 0) | q[count + 2]
                else:
                    Sgate(SMSV_r, np.pi) | q[count + 2]
                count += 1            
        else:
            raise ValueError('SMSV_phi pattern unrecognized')
        T_indiv = T_wire ** (1 / (num_E + num_M2))
        for i in range(1, num_M2 + num_E + 1):
            for ii in range(2 + num_E + num_M2):
                ThermalLossChannel(T_indiv, nbar) | q[ii] # Wire loss
            for ii in range(i, 0, -1):
                ThermalLossChannel(T_bs ** .5, nbar) | q[ii] # Beam splitter loss
                ThermalLossChannel(T_bs ** .5, nbar) | q[ii + 1] # Beam splitter loss
                BSgate(BS_theta) | (q[ii], q[ii + 1])
                ThermalLossChannel(T_bs ** .5, nbar) | q[ii] # Beam splitter loss
                ThermalLossChannel(T_bs ** .5, nbar) | q[ii + 1] # Beam splitter loss
        for j in range(2 + num_E + num_M2):
            ThermalLossChannel(T_end_face, nbar) | q[j] # End-face loss
        for j in range(2 + num_E + num_M2):
            ThermalLossChannel(T_detector, nbar) | q[j] # Detector loss
    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    return result.state.cov()