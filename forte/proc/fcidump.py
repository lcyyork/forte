#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2019 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
"""Module with utility function for dumping the Hamiltonian to file in FCIDUMP format."""

from datetime import datetime

import numpy as np

from psi4.driver import psifiles as psif
from psi4.driver.procrouting.proc_util import check_iwl_file_from_scf_type

from psi4 import core


def fcidump(wfn, fname='INTDUMP', oe_ints=None, write_pntgrp=False):
    """Save integrals to file in FCIDUMP format as defined in Comp. Phys. Commun. 54 75 (1989)
    Additional one-electron integrals, including orbital energies, can also be saved.
    This latter format can be used with the HANDE QMC code but is not standard.
    This function converts the irrep labels from Cotton's order (used in psi4)
    to molpro ordering. Since the latter is ambiguous unless the point group is
    specificied, this function has an option to write the point group label in the
    FCIDUMP file. This, however, is not part of the standard.

    :returns: None

    :raises: ValidationError when SCF wavefunction is not RHF, ROHF, or UHF

    :type wfn: :py:class:`~psi4.core.Wavefunction`
    :param wfn: set of molecule, basis, orbitals from which to generate cube files
    :param fname: name of the integrals file, defaults to INTDUMP
    :param oe_ints: list of additional one-electron integrals to save to file.
    So far only EIGENVALUES is a valid option.
    :param write_pntgrp: write the point group to file. This information is not
    part of the Comp Phys Chem standard and might create issues with parsing for codes
    that do not support it.

    :examples:

    >>> # [1] Save one- and two-electron integrals to standard FCIDUMP format
    >>> E, wfn = energy('scf', return_wfn=True)
    >>> fcidump(wfn)

    >>> # [2] Save orbital energies, one- and two-electron integrals.
    >>> E, wfn = energy('scf', return_wfn=True)
    >>> fcidump(wfn, oe_ints=['EIGENVALUES'])

    """
    # Get some options
    reference = core.get_option('SCF', 'REFERENCE')
    ints_tolerance = core.get_global_option('INTS_TOLERANCE')
    # Some sanity checks
    if reference not in ['RHF', 'UHF', 'ROHF']:
        raise ValidationError('FCIDUMP not implemented for {} references\n'.format(reference))
    if oe_ints is None:
        oe_ints = []

    molecule = wfn.molecule()
    docc = wfn.doccpi()
    frzcpi = wfn.frzcpi()
    frzvpi = wfn.frzvpi()
    active_docc = docc - frzcpi
    active_socc = wfn.soccpi()
    active_mopi = wfn.nmopi() - frzcpi - frzvpi

    nbf = active_mopi.sum() if wfn.same_a_b_orbs() else 2 * active_mopi.sum()
    nirrep = wfn.nirrep()
    nelectron = 2 * active_docc.sum() + active_socc.sum()
    symm = wfn.molecule().point_group().symbol()
    irrep_map = _irrep_map(symm)

    wfn_irrep = 0
    for h, n_socc in enumerate(active_socc):
        if n_socc % 2 == 1:
            wfn_irrep ^= h

    core.print_out('Writing integrals in FCIDUMP format to ' + fname + '\n')
    # Generate FCIDUMP header
    header = '&FCI\n'
    header += 'NORB={:d},\n'.format(nbf)
    header += 'NELEC={:d},\n'.format(nelectron)
    header += 'MS2={:d},\n'.format(wfn.nalpha() - wfn.nbeta())
    header += 'UHF=.{}.,\n'.format(not wfn.same_a_b_orbs()).upper()
    orbsym = ''
    for h in range(active_mopi.n()):
        for n in range(frzcpi[h], frzcpi[h] + active_mopi[h]):
            orbsym += '{:d},'.format(irrep_map[h])
            if not wfn.same_a_b_orbs():
                orbsym += '{:d},'.format(irrep_map[h])
    header += 'ORBSYM={}\n'.format(orbsym)
    header += 'ISYM={:d},\n'.format(irrep_map[wfn_irrep])
    if write_pntgrp:
        header += 'PNTGRP={},\n'.format(symm.upper())
    header += '&END\n'
    with open(fname, 'w') as intdump:
        intdump.write(header)

    # Get an IntegralTransform object
    check_iwl_file_from_scf_type(core.get_global_option('SCF_TYPE'), wfn)
    spaces = [core.MOSpace.all()]
    trans_type = core.IntegralTransform.TransformationType.Restricted
    if not wfn.same_a_b_orbs():
        trans_type = core.IntegralTransform.TransformationType.Unrestricted
    ints = core.IntegralTransform(wfn, spaces, trans_type)
    ints.transform_tei(core.MOSpace.all(), core.MOSpace.all(), core.MOSpace.all(), core.MOSpace.all())
    core.print_out('Integral transformation complete!\n')

    DPD_info = {'instance_id': ints.get_dpd_id(), 'alpha_MO': ints.DPD_ID('[A>=A]+'), 'beta_MO': 0}
    if not wfn.same_a_b_orbs():
        DPD_info['beta_MO'] = ints.DPD_ID("[a>=a]+")
    # Write TEI to fname in FCIDUMP format
    core.fcidump_tei_helper(nirrep, wfn.same_a_b_orbs(), DPD_info, ints_tolerance, fname)

    # Read-in OEI and write them to fname in FCIDUMP format
    # Indexing functions to translate from zero-based (C and Python) to
    # one-based (Fortran)
    mo_idx = lambda x: x + 1
    alpha_mo_idx = lambda x: 2 * x + 1
    beta_mo_idx = lambda x: 2 * (x + 1)

    with open(fname, 'a') as intdump:
        core.print_out('Writing frozen core operator in FCIDUMP format to ' + fname + '\n')
        if reference == 'RHF' or reference == 'ROHF':
            PSIF_MO_FZC = 'MO-basis Frozen-Core Operator'
            moH = core.Matrix(PSIF_MO_FZC, wfn.nmopi(), wfn.nmopi())
            moH.load(core.IO.shared_object(), psif.PSIF_OEI)
            mo_slice = core.Slice(frzcpi, active_mopi)
            MO_FZC = moH.get_block(mo_slice, mo_slice)
            offset = 0
            for h, block in enumerate(MO_FZC.nph):
                il = np.tril_indices(block.shape[0])
                for index, x in np.ndenumerate(block[il]):
                    row = mo_idx(il[0][index] + offset)
                    col = mo_idx(il[1][index] + offset)
                    if (abs(x) > ints_tolerance):
                        intdump.write('{:29.20E} {:4d} {:4d} {:4d} {:4d}\n'.format(x, row, col, 0, 0))
                offset += block.shape[0]
            # Additional one-electron integrals as requested in oe_ints
            # Orbital energies
            core.print_out('Writing orbital energies in FCIDUMP format to ' + fname + '\n')
            if 'EIGENVALUES' in oe_ints:
                eigs_dump = write_eigenvalues(wfn.epsilon_a().get_block(mo_slice).to_array(), mo_idx)
                intdump.write(eigs_dump)
        else:
            PSIF_MO_A_FZC = 'MO-basis Alpha Frozen-Core Oper'
            moH_A = core.Matrix(PSIF_MO_A_FZC, wfn.nmopi(), wfn.nmopi())
            moH_A.load(core.IO.shared_object(), psif.PSIF_OEI)
            mo_slice = core.Slice(frzcpi, active_mopi)
            MO_FZC_A = moH_A.get_block(mo_slice, mo_slice)
            offset = 0
            for h, block in enumerate(MO_FZC_A.nph):
                il = np.tril_indices(block.shape[0])
                for index, x in np.ndenumerate(block[il]):
                    row = alpha_mo_idx(il[0][index] + offset)
                    col = alpha_mo_idx(il[1][index] + offset)
                    if (abs(x) > ints_tolerance):
                        intdump.write('{:29.20E} {:4d} {:4d} {:4d} {:4d}\n'.format(x, row, col, 0, 0))
                offset += block.shape[0]
            PSIF_MO_B_FZC = 'MO-basis Beta Frozen-Core Oper'
            moH_B = core.Matrix(PSIF_MO_B_FZC, wfn.nmopi(), wfn.nmopi())
            moH_B.load(core.IO.shared_object(), psif.PSIF_OEI)
            mo_slice = core.Slice(frzcpi, active_mopi)
            MO_FZC_B = moH_B.get_block(mo_slice, mo_slice)
            offset = 0
            for h, block in enumerate(MO_FZC_B.nph):
                il = np.tril_indices(block.shape[0])
                for index, x in np.ndenumerate(block[il]):
                    row = beta_mo_idx(il[0][index] + offset)
                    col = beta_mo_idx(il[1][index] + offset)
                    if (abs(x) > ints_tolerance):
                        intdump.write('{:29.20E} {:4d} {:4d} {:4d} {:4d}\n'.format(x, row, col, 0, 0))
                offset += block.shape[0]
            # Additional one-electron integrals as requested in oe_ints
            # Orbital energies
            core.print_out('Writing orbital energies in FCIDUMP format to ' + fname + '\n')
            if 'EIGENVALUES' in oe_ints:
                alpha_eigs_dump = write_eigenvalues(wfn.epsilon_a().get_block(mo_slice).to_array(), alpha_mo_idx)
                beta_eigs_dump = write_eigenvalues(wfn.epsilon_b().get_block(mo_slice).to_array(), beta_mo_idx)
                intdump.write(alpha_eigs_dump + beta_eigs_dump)
        # Dipole integrals
        #core.print_out('Writing dipole moment OEI in FCIDUMP format to ' + fname + '\n')
        # Traceless quadrupole integrals
        #core.print_out('Writing traceless quadrupole moment OEI in FCIDUMP format to ' + fname + '\n')
        # Frozen core + nuclear repulsion energy
        core.print_out('Writing frozen core + nuclear repulsion energy in FCIDUMP format to ' + fname + '\n')
        e_fzc = ints.get_frozen_core_energy()
        e_nuc = molecule.nuclear_repulsion_energy(wfn.get_dipole_field_strength())
        intdump.write('{: 29.20E} {:4d} {:4d} {:4d} {:4d}\n'.format(e_fzc + e_nuc, 0, 0, 0, 0))
    core.print_out('Done generating {} with integrals in FCIDUMP format.\n'.format(fname))


def write_eigenvalues(eigs, mo_idx):
    """Prepare multi-line string with one-particle eigenvalues to be written to the FCIDUMP file.
    """
    eigs_dump = ''
    iorb = 0
    for h, block in enumerate(eigs):
        for idx, x in np.ndenumerate(block):
            eigs_dump += '{: 29.20E} {:4d} {:4d} {:4d} {:4d}\n'.format(x, mo_idx(iorb), 0, 0, 0)
            iorb += 1
    return eigs_dump


def _irrep_map(symm):
    """Returns an array of irrep indices that maps from Psi4's ordering convention to the standard FCIDUMP convention.
    """
    psi2dump = {'c1' : [1],               # A
                'ci' : [1,2],             # Ag Au
                'c2' : [1,2],             # A  B
                'cs' : [1,2],             # A' A"
                'd2' : [1,4,3,2],         # A  B1  B2  B3
                'c2v' : [1,4,2,3],        # A1 A2  B1  B2
                'c2h' : [1,4,2,3],        # Ag Bg  Au  Bu
                'd2h' : [1,4,6,7,8,5,3,2] # Ag B1g B2g B3g Au B1u B2u B3u
                }

    irrep_map = psi2dump[symm.lower()]
    return np.array(irrep_map, dtype='int')


def _irrep_map_inverse(symm):
    """Returns an array of irrep indices that maps from the standard FCIDUMP convention to the Psi4's ordering convention.
    """
    dump2psi = {'c1' : [-1,0],               # A
                'ci' : [-1,0,1],             # Ag Au
                'c2' : [-1,0,1],             # A  B
                'cs' : [-1,0,1],             # A' A"
                'd2' : [-1,0,3,2,1],         # A  B1  B2  B3
                'c2v' : [-1,0,2,3,1],        # A1 A2  B1  B2
                'c2h' : [-1,0,2,3,1],        # Ag Bg  Au  Bu
                'd2h' : [-1,0,7,6,1,5,2,3,4] # Ag B1g B2g B3g Au B1u B2u B3u
                }

    irrep_map = dump2psi[symm.lower()]
    return np.array(irrep_map, dtype='int')


def fcidump_from_file(fname, convert_to_psi4=False):
    """Function to read in a FCIDUMP file.

    :returns: a dictionary with FCIDUMP header and integrals
    The key-value pairs are:
      - 'norb' : number of basis functions
      - 'nelec' : number of electrons
      - 'ms2' : spin polarization of the system
      - 'isym' : symmetry of state (if present in FCIDUMP)
      - 'pntgrp' : point group (if present in FCIDUMP)
      - 'orbsym' : list of symmetry labels of each orbital
      - 'uhf' : whether restricted or unrestricted
      - 'enuc' : nuclear repulsion plus frozen core energy
      - 'epsilon' : orbital energies
      - 'hcore' : core Hamiltonian
      - 'eri' : electron-repulsion integrals

    :param fname: FCIDUMP file name
    :param convert_to_psi4: If turned on and the FCIDUMP
    file contains the PNTGRP label, the orbital symmetries will
    be converted to the ordering used in psi4
    """
    intdump = {}
    with open(fname, 'r') as handle:
        assert '&FCI' == handle.readline().strip()

        skiplines = 1
        read = True
        while True:
            skiplines += 1
            line = handle.readline()
            if 'END' in line:
                break

            key, value = line.split('=')
            value = value.strip().rstrip(',')
            if key == 'UHF':
                value = 'TRUE' in value
            elif key == 'ORBSYM':
                value = [int(x) for x in value.split(',')]                
            elif key == 'PNTGRP':
                pass
            else:
                value = int(value.replace(',', ''))

            intdump[key.lower()] = value

    if convert_to_psi4 and ('pntgrp' in intdump) and ('orbsym' in intdump):
        irrep_map_inverse = _irrep_map_inverse(intdump['pntgrp'])
        psi4_irrep_map = map(lambda x: irrep_map_inverse[x], intdump['orbsym'])
        intdump['orbsym'] = list(psi4_irrep_map)
        intdump['isym'] = irrep_map_inverse[intdump['isym']]

    # Read the data and index, skip header
    raw_ints = np.genfromtxt(fname, skip_header=skiplines)

    # Read last line, i.e. Enuc + Efzc
    intdump['enuc'] = raw_ints[-1, 0]

    # Read in integrals and indices
    ints = raw_ints[:-1, 0]

    # Get dimensions and indices
    nbf = intdump['norb']
    idxs = raw_ints[:, 1:].astype(int) - 1

    # Slices
    sl = slice(ints.shape[0] - nbf, ints.shape[0])

    # Count how many 1-index intdump we have
    one_index = np.all(idxs[sl, 1:] == -1, axis=1).sum()

    # Extract orbital energies if present
    if one_index > 0:
        epsilon = np.zeros(nbf)
        epsilon[idxs[sl, 0]] = ints[sl]
        intdump['epsilon'] = epsilon

    # Count how many 2-index intdump we have
    sl = slice(ints.shape[0] - one_index - nbf * nbf, sl.stop - one_index)
    two_index = np.all(idxs[sl, 2:] == -1, axis=1).sum()
    sl = slice(ints.shape[0] - two_index - one_index, ints.shape[0] - one_index)

    # Extract Hcore
    Hcore = np.zeros((nbf, nbf))
    Hcore[(idxs[sl, 0], idxs[sl, 1])] = ints[sl]
    Hcore[(idxs[sl, 1], idxs[sl, 0])] = ints[sl]
    intdump['hcore'] = Hcore

    # Extract ERIs
    sl = slice(0, sl.start)
    eri = np.zeros((nbf, nbf, nbf, nbf))
    eri[(idxs[sl, 0], idxs[sl, 1], idxs[sl, 2], idxs[sl, 3])] = ints[sl]
    eri[(idxs[sl, 0], idxs[sl, 1], idxs[sl, 3], idxs[sl, 2])] = ints[sl]
    eri[(idxs[sl, 1], idxs[sl, 0], idxs[sl, 2], idxs[sl, 3])] = ints[sl]
    eri[(idxs[sl, 1], idxs[sl, 0], idxs[sl, 3], idxs[sl, 2])] = ints[sl]
    eri[(idxs[sl, 2], idxs[sl, 3], idxs[sl, 0], idxs[sl, 1])] = ints[sl]
    eri[(idxs[sl, 3], idxs[sl, 2], idxs[sl, 0], idxs[sl, 1])] = ints[sl]
    eri[(idxs[sl, 2], idxs[sl, 3], idxs[sl, 1], idxs[sl, 0])] = ints[sl]
    eri[(idxs[sl, 3], idxs[sl, 2], idxs[sl, 1], idxs[sl, 0])] = ints[sl]
    intdump['eri'] = eri

    return intdump
