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
thistFile: {THFILE}
logFile: {CKFILE}
fuel_mole_fracs: {{ {FUEL_FRACS} }}
oxidizer_mole_fracs: {{ {OXID_FRACS} }}
trace_mole_fracs: {{ {TRACE_FRACS} }} #this is just for printing
delta_T_ignition: 400.0
stop_time: 1.0
print_time: 1.0
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
one_step_mode: 1
print_net_rates_of_progress: 1
continue_after_ignition: 0
"""

ZERORK_EXE=os.getenv("ZERORK_EXE", default='/usr/apps/advcomb/bin/constVolumeWSR_yml.x')

def zerork(dir_desk, atm, T0, fuel_fracs, oxid_fracs, phi, species_names, rxn_equations, eps=0.05, dir_raw=None):
    cpu0 = time.time()

    print('>'*30)
    print('zerork for phi='+ str(phi) + ' at '+ str(atm)+'atm' + ' and '+str(T0)+'K')
    print('<'*30)
    
    p = ct.one_atm * atm

    fuel_fracs_str = ','.join([str(k)+": " + str(v) for k,v in fuel_fracs.items()])
    oxid_fracs_str = ','.join([str(k)+": " + str(v) for k,v in oxid_fracs.items()])
    other_species = []
    for sp in species_names:
        if sp not in fuel_fracs and sp not in oxid_fracs:
            other_species.append(sp)
    trace_fracs_str = ','.join([str(sp)+": 0" for sp in other_species])

    #Write zero-rk input file
    error_return = False
    try:
        tmpdir = tempfile.mkdtemp(dir=dir_desk)
        zerork_out_file=open(os.path.join(tmpdir,'zerork.out'),'a')
        zerork_out_file.write('!!! Running ZeroRK !!!\n')
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
                TRACE_FRACS=trace_fracs_str))

        zerork_out=''
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
                                                stderr=subprocess.STDOUT,universal_newlines=True).split('\n')
        except subprocess.CalledProcessError as e:
            zerork_out_file.write('!!! Warning: ZeroRK exited with non-zero output ({}).\n'.format(e.returncode))
            zerork_out=e.output.split('\n')
            error_return = True

        for line in zerork_out:
            zerork_out_file.write(line+'\n')

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
        try:
            with open(os.path.join(tmpdir,'zerork.thist'),'r') as datfile:
                for line in datfile:
                    if len(line) <= 1:
                        if start_data: break #done with first block break out
                        start_data = False
                        continue
                    if line[0] != '#':
                        start_data = True
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
                        nsp_log = len(calc_species[0])
                        vals = list(map(float,line.split()))
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

        except IOError:
            print("No data file from ZeroRK, ZeroRK output was:")
            for line in zerork_out:
                print("\t", line)
            sys.exit()

        zerork_out_file.close()

    #Clean up
    finally:
        shutil.rmtree(tmpdir)
        pass

    if(error_return or len(raw['axis0']) == 0):
        print(f"Zero-rk failed: {atm}, {T0}, {phi}")
        sys.exit()

    raw['net_reaction_rate'] = np.matrix(raw['net_reaction_rate']) * 1.0e3 #convert to mol/m^3/s
    raw['mole_fraction'] = np.matrix(raw['mole_fraction'])

    ign_delay = raw['axis0'][-1]
    rdp_array = [ np.array([x,y]) for x,y in zip(raw['axis0'],raw['temperature']) ]
    resampled = rdp(rdp_array, epsilon=eps*ign_delay)

    #resample_time = np.linspace(0, ign_delay, 100)
    resample_time = np.array([ x[0] for x in resampled ])
    for key in ['temperature', 'pressure', 'volume', 'mole', 'heat_release_rate']:
        out = np.interp(resample_time, raw['axis0'], raw[key])
        raw[key] = out
    for key in ['net_reaction_rate', 'mole_fraction']:
        new_mat = np.zeros( (resample_time.shape[0], raw[key].shape[1]) )
        for col in range(raw[key].shape[1]):
            new_mat[:,col] = np.interp(resample_time, raw['axis0'], raw[key][:,col].flat)
        raw[key] = new_mat
    raw['axis0'] = resample_time

    print('n_points = ' + str(len(raw['axis0'])))
    print('CPU time = '+str(time.time() - cpu0))

    if dir_raw is not None:
        path_raw = os.path.join(dir_raw,'raw.npz')
        raw = save_raw_npz(raw, path_raw)
        save_raw_csv(raw, species_names, rxn_equations, dir_raw)

    return raw


def test_senkin():

    #from src.inp.inp_TRF import dir_public, fuel_dict
    #soln = ct.Solution(os.path.join(dir_public,'detailed','mech','chem.cti'))
    soln = ct.Solution('gri30.xml')

    atm = 1
    T0 = 1000.0
    phi = 1

    X0 = 'CH4:1, O2:2, N2:7.52'
    dir_raw = 'test'
    raw = senkin(soln, atm, T0, X0, if_half=True, if_fine=False, dir_raw=dir_raw)
    #raw = save_raw(raw, None)

    tt = raw['axis0']
    TT = raw['temperature']
    qq = raw['heat_release_rate']
    #print(str(len(tt)) + ' points')
    #print(raw['net_reaction_rate'].shape)

    plt.plot(qq, TT, marker='o')
    #plt.savefig(os.path.join(dir_raw,'ign_fine.jpg'))
    plt.show()

    #print(tt[-1])

    #return raw


if __name__=="__main__":
    test_senkin()
    #raw = load_raw('raw.npz')
    #plt.plot(raw['axis0'],raw['temperature'])
    #plt.show()
