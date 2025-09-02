import os
import sys
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from .def_zk_tools import *
import copy
import os
import time
import tempfile
import subprocess
import shutil

BASE_YML="""
mechFile: {MECHDIR}/chem.inp
thermFile: {MECHDIR}/therm.dat
idtFile: {IDTFILE}
thistFile: /dev/null
logFile: {CKFILE}
fuel_mole_fracs: {{ {FUEL_FRACS} }}
oxidizer_mole_fracs: {{ {OXID_FRACS} }}
trace_mole_fracs: {{ {TRACE_FRACS} }} #this is just for printing
temperature_deltas: {TEMPERATURE_DELTAS}
temperature_print_resolution: {PRINT_RESOLUTION}
stop_time: 10.0
print_time: 10.0
relative_tolerance: 1.0e-8
absolute_tolerance: 1.0e-20
initial_temperatures: [ {TEMP} ]
initial_pressures: [ {PRES} ]
initial_phis: [ {PHI} ]
initial_egrs: [ 0.0 ]
preconditioner_thresholds: [ 2.048e-3 ]
eps_lin: 0.05
nonlinear_convergence_coefficient: 0.05
long_output: 1
one_step_mode: 0
print_net_rates_of_progress: 1
continue_after_ignition: 0
"""

BASE_PSR_YML="""
mechFile: {MECHDIR}/chem.inp
thermFile: {MECHDIR}/therm.dat
idtFile: {IDTFILE}
#thistFile: {THFILE}
thistFile: /dev/null
logFile: {CKFILE}
stop_time: {ENDTIME}
print_time: {ENDTIME}
relative_tolerance: 1.0e-8
absolute_tolerance: 1.0e-18
initial_temperatures: [ {TEMP} ]
initial_pressures: [ {PRES} ]
initial_phis: [ {PHI} ]
initial_egrs: [ 0.0 ]
residence_times: [ {TAU} ]
preconditioner_thresholds: [ 1.0e-6 ]
eps_lin: 0.05
nonlinear_convergence_coefficient: 0.05
long_output: 1
print_net_rates_of_progress: 1
fuel_mole_fracs: {{ {FUEL_FRACS} }}
oxidizer_mole_fracs: {{ {OXID_FRACS} }}
trace_mole_fracs: {{ {TRACE_FRACS} }} #this is just for printing
full_mole_fracs: {{ {FULL_FRACS} }} 
pressure_controller_coefficient: 1.0
use_equilibrium_for_initialization: n
ignition_temperature: 2500
"""

ZERORK_HOME=os.getenv("ZERORK_HOME", default='/usr/apps/advcomb')
ZERORK_EXE=os.path.join(ZERORK_HOME, "bin", "constVolumeWSR.x")
ZERORK_PSR_EXE=os.path.join(ZERORK_HOME, "bin", "constVolumePSR.x")

def read_zerork_outfile(zerork_out):
    start_data=False
    calc_species = []
    raw = dict()
    raw['axis0'] = []
    raw['axis0_type'] = 'time'
    raw['pressure'] = []
    raw['temperature'] = []
    raw['volume'] = []
    raw['mole_fraction'] = []
    raw['net_reaction_rate'] = []
    raw['mole'] = []
    raw['heat_release'] = []
    raw['heat_release_rate'] = []
    #TODO: Parsing for sweeps (i.e. run_id != 0)
    nrxn = 0
    for line in zerork_out:
        if len(line) <= 1:
            if start_data: break #done with first block break out
            start_data = False
            continue
        if line[0] != '#':
            start_data = True
            nsp_log = len(calc_species[0])
        if "run id" in line:
            tokens = line.split()
            tmp_list = []
            for i,tok in enumerate(tokens):
                if tok == "mlfrc":
                    tmp_list.append(tokens[i+1])
                if tok == "rop":
                    nrxn += 1
            if len(tmp_list) > 0:
                calc_species.append(tmp_list)
        if start_data:
            try:
                vals = list(map(float,line.split()))
            except Exception as e:
                raise e
            if(len(vals) < 8): continue

            raw['axis0'].append(vals[1])
            raw['temperature'].append(vals[2])
            raw['pressure'].append(vals[3])
            raw['volume'].append(1/vals[4])
            raw['mole_fraction'].append(vals[8:8+nsp_log])
            raw['net_reaction_rate'].append(vals[8+nsp_log:8+nsp_log+nrxn])
            raw['mole'].append(vals[5]/vals[6]*1e3) #density / molecular weight => inverse molar volume
            raw['heat_release'].append(vals[8])
            if len(raw['axis0']) > 1:
                hrr = -(raw['heat_release'][-1] - raw['heat_release'][-2])
                hrr /= raw['axis0'][-1] - raw['axis0'][-2]
                hrr /= vals[4] #volumetric heat release
                raw['heat_release_rate'].append(hrr)
            else:
                raw['heat_release_rate'].append(0)

    raw['net_reaction_rate'] = np.matrix(raw['net_reaction_rate']) * 1.0e3 #convert to mol/m^3/s
    raw['mole_fraction'] = np.matrix(raw['mole_fraction'])

    return raw

def zerork(dir_desk, atm, T0, fuel_fracs, oxid_fracs, phi, species_names, rxn_equations, is_fine=False, dir_raw=None):
    cpu0 = time.time()

    print('>'*30)
    print('zerork for phi='+ str(phi) + ' at '+ str(atm)+'atm' + ' and '+str(T0)+'K')
    print('<'*30)
    
    p = ct.one_atm * atm

    fuel_fracs_str = ','.join([f'"{k}": ' + str(v) for k,v in fuel_fracs.items()])
    oxid_fracs_str = ','.join([f'"{k}": ' + str(v) for k,v in oxid_fracs.items()])
    other_species = []
    for sp in species_names:
        if sp not in fuel_fracs and sp not in oxid_fracs:
            other_species.append(sp)
    trace_fracs_str = ',\n'.join([f'"{sp}": 0' for sp in other_species])

    temperature_deltas="[2, 5, 10, 15, 20, 25, 50, 75, 100, 200, 300, 400.0]"
    print_resolution='1000.0'
    if(is_fine):
        temperature_deltas="[400.0]"
        print_resolution='2.0'
    #Write zero-rk input file
    error_return = False
    try:
        tmpdir = tempfile.mkdtemp(dir=dir_desk)
        zerork_infile_name = os.path.join(tmpdir,'zerork.yml')
        with open(zerork_infile_name,'w') as infile:
            infile.write(BASE_YML.format(
                MECHDIR=os.path.join(dir_desk, 'mech'),
                CKFILE=os.path.join(tmpdir,'zerork.cklog'),
                IDTFILE=os.path.join(tmpdir,'zerork.dat'),
                THFILE=os.path.join(tmpdir,'zerork.thist'),
                TEMP=T0,
                PRES=p,
                PHI=phi,
                OXID_FRACS=oxid_fracs_str,
                FUEL_FRACS=fuel_fracs_str,
                TRACE_FRACS=trace_fracs_str,
                TEMPERATURE_DELTAS=temperature_deltas,
                PRINT_RESOLUTION=print_resolution))

        zerork_out=[]
        env = dict(os.environ)
        env["ZERORK_SPLIT_REVERSIBLE_REACTIONS"] = str(1)
        try:
            #if('mpi_procs' in params and params['mpi_procs'] > 1 and self.zerork_mpi_exe):
            #    np=str(params['mpi_procs'])
            #    mpi_cmd = params.get('mpi_cmd','srun -n')
            #    if(mpi_cmd == 'mpirun') : mpi_cmd += ' -np'
            #    if(mpi_cmd == 'srun') : mpi_cmd += ' -n'
            #    cmd_list = params['mpi_cmd'].split() + [np,self.zerork_mpi_exe,zerork_infile_name]
            #    zerork_out=subprocess.check_output(cmd_list, stderr=subprocess.STDOUT,
            #                                       universal_newlines=True).split('\n')
            #else:
            zerork_out=subprocess.check_output([ZERORK_EXE,zerork_infile_name],
                                                stderr=subprocess.STDOUT,universal_newlines=True, env=env).split('\n')
            raw = read_zerork_outfile(zerork_out)
            shutil.rmtree(tmpdir)

        except subprocess.CalledProcessError as e:
            zerork_out_file=open(os.path.join(tmpdir,'zerork.out'),'a')
            zerork_out_file.write('!!! Running ZeroRK !!!\n')
            zerork_out_file.write('!!! Warning: ZeroRK exited with non-zero output ({}).\n'.format(e.returncode))
            zerork_out=e.output.split('\n')
            error_return = True

            for line in zerork_out:
                zerork_out_file.write(line+'\n')

            zerork_out_file.close()


    except Exception as e:
       raise e
    #Clean up
    finally:
        pass

    if(error_return or len(raw['axis0']) == 0):
        print(f"Zero-rk failed: {atm}, {T0}, {phi}")
        raise ValueError

    print('n_points = ' + str(len(raw['axis0'])))
    print('CPU time = '+str(time.time() - cpu0))

    if dir_raw is not None:
        path_raw = os.path.join(dir_raw,'raw.npz')
        raw = save_raw_npz(raw, path_raw)
        save_raw_csv(raw, species_names, rxn_equations, dir_raw)

    return raw


def zerork_psr(dir_desk, tau, p, T0, fuel_fracs, oxid_fracs, phi, species_names, rxn_equations, full_fracs=None):
    cpu0 = time.time()
    raw = None

    fuel_fracs_str = ','.join([f'"{k}": ' + str(v) for k,v in fuel_fracs.items()])
    oxid_fracs_str = ','.join([f'"{k}": ' + str(v) for k,v in oxid_fracs.items()])
    other_species = []
    for sp in species_names:
        if sp not in fuel_fracs and sp not in oxid_fracs:
            other_species.append(sp)
    trace_fracs_str = ',\n'.join([f'"{sp}": 0' for sp in other_species])
    full_fracs_str = ""
    if(full_fracs is not None):
        full_fracs_str = ',\n'.join(['"{k}": {v}' for k,v in full_fracs.items()])

    #Write zero-rk input file
    error_return = False
    try:
        tmpdir = tempfile.mkdtemp(dir=dir_desk)
        zerork_infile_name = os.path.join(tmpdir,'zerork.yml')
        with open(zerork_infile_name,'w') as infile:
            infile.write(BASE_PSR_YML.format(
                MECHDIR=os.path.join(dir_desk, 'mech'),
                CKFILE=os.path.join(tmpdir,'zerork.cklog'),
                IDTFILE=os.path.join(tmpdir,'zerork.dat'),
                THFILE=os.path.join(tmpdir,'zerork.thist'),
                TEMP=T0,
                PRES=p,
                PHI=phi,
                TAU=tau,
                ENDTIME=200*tau,
                OXID_FRACS=oxid_fracs_str,
                FUEL_FRACS=fuel_fracs_str,
                TRACE_FRACS=trace_fracs_str,
                FULL_FRACS=full_fracs_str))

        zerork_out=[]
        env = dict(os.environ)
        env["ZERORK_SPLIT_REVERSIBLE_REACTIONS"] = str(1)
        try:
            zerork_out=subprocess.check_output([ZERORK_PSR_EXE,zerork_infile_name],
                                                stderr=subprocess.DEVNULL,universal_newlines=True, env=env).split('\n')
            shutil.rmtree(tmpdir)

        except subprocess.CalledProcessError as e:
            zerork_out_file=open(os.path.join(tmpdir,'zerork.out'),'a')
            zerork_out_file.write('!!! Running ZeroRK !!!\n')
            zerork_out_file.write('!!! Warning: ZeroRK exited with non-zero output ({}).\n'.format(e.returncode))
            zerork_out=e.output.split('\n')
            error_return = True

            for line in zerork_out:
                zerork_out_file.write(line+'\n')

            zerork_out_file.close()

        try:
             full_raw = read_zerork_outfile(zerork_out)
             #full_raw is time series.  For PSR we just want the final point
             raw = dict()
             raw['axis0'] = [tau]
             raw['axis0_type'] = 'residence_time'
             for k in ['pressure', 'temperature', 'volume', 'mole', 'mole_fraction',
                       'net_reaction_rate', 'heat_release', 'heat_release_rate']:
                 raw[k] = full_raw[k][-1:]
        except:
            pass


    #Clean up
    finally:
        pass
        #shutil.rmtree(tmpdir)

    #if(error_return or len(raw['axis0']) == 0):
    #    print(f"Zero-rk failed: {p}, {T0}, {phi}")
    #    raise ValueError

    if(raw):
        print('n_points = ' + str(len(raw['axis0'])))
    else:
        print("No data from Zero-RK")
    print('CPU time = '+str(time.time() - cpu0))
    return raw


def zerork_S_curve(dir_desk, atm, T0, fuel_fracs, oxid_fracs, phi, species_names, rxn_equations, eps=0.005, dir_raw=None):

    print('>'*30)
    print('zerork S-curve for phi='+ str(phi) + ' at '+ str(atm)+'atm' + ' and '+str(T0)+'K')
    print('<'*30)
    
    if len(species_names) > 100:
        verbose = True
    else:
        verbose = False

    p = ct.one_atm * atm

    tau = 1
    raw_burn = None
    T_burn = None
    tau_burn = tau

    tau_r = 0.5

    while True:
        print(f'tau: phi={phi} at {atm} atm and {T0} K : {tau}, {tau_r}')
        raw = zerork_psr(dir_desk, tau, p, T0, fuel_fracs, oxid_fracs, phi, species_names, rxn_equations)
        if raw is None:
            T = T0
        else: 
            T = raw["temperature"][-1]

        print(f'T: phi={phi} at {atm} atm and {T0} K : {T}, {T_burn}')
        if T_burn is None:
            T_burn = T

        if abs(T_burn - T) > 50:
            # if extinction happens or dT too large
            if tau_r > 0.999:
                if verbose:            
                    print('finished, tau_r = '+str(tau_r))
                break
            else:
                tau = tau_burn
                tau_r = tau_r + (1-tau_r)*0.5
                if verbose:            
                    print('refined, tau_r = '+str(tau_r))
        else:
            T_burn = T
            tau_burn = tau
            if raw_burn is None:
                raw_burn = raw
            else:
                for k in ['axis0', 'pressure', 'temperature', 'volume', 'mole',
                          'heat_release', 'heat_release_rate']:
                    raw_burn[k].append(raw[k][-1])
                for k in ['mole_fraction', 'net_reaction_rate']:
                    raw_burn[k] = np.vstack([raw_burn[k],raw[k][-1]])
            save_raw_npz(raw, os.path.join(dir_raw,'raw_temp.npz'))

        tau *= tau_r

    resample = True
    if(resample):
        rdp_array = [ np.array([np.log(x),y/1000]) for x,y in zip(np.flip(raw_burn['axis0']),np.flip(raw_burn['temperature'])) ]
        resampled = rdp(rdp_array, epsilon=eps)

        resample_tau = np.array([ np.exp(x[0]) for x in resampled ])
        for key in ['temperature', 'pressure', 'volume', 'mole', 'heat_release_rate']:
            out = np.interp(resample_tau, np.flip(raw_burn['axis0']), np.flip(raw_burn[key]))
            raw_burn[key] = np.flip(out)
        for key in ['net_reaction_rate', 'mole_fraction']:
            new_mat = np.zeros( (resample_tau.shape[0], raw_burn[key].shape[1]) )
            for col in range(raw_burn[key].shape[1]):
                new_mat[:,col] = np.flip(np.interp(resample_tau,
                                                   np.flip(raw_burn['axis0']),
                                                   np.flip(raw_burn[key][:,col].flat)))
            raw_burn[key] = new_mat
        raw_burn['axis0'] = np.flip(resample_tau)

    raw_burn = save_raw_npz(raw_burn, os.path.join(dir_raw,'raw.npz'))
    save_raw_csv(raw_burn, species_names, rxn_equations, dir_raw)

    return raw

if __name__=="__main__":
    pass
    #raw = load_raw('raw.npz')
    #plt.plot(raw['axis0'],raw['temperature'])
    #plt.show()
