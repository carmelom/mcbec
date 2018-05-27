'''Main module for mcbec.
'''

import time

__all__ = ['canonical_sampling']


def canonical_sampling(P, ID, beta=1e-2,
                        n_ensamble=1e4, n_max=4e4):
    '''
    General-purpose sampling of a canonical ensamble.

    Parameters
    ----------
    P : object
        Instance of a custom class, wich includes attributes

        + P.beta
        + P.energy
        + P.state
        + P.step

        and methods:

        + P.set_beta(beta)
        + P.MC_move(), returning 1/0 (accepted/rejected)
        + P.update_MC_parameters(acc_ratio)

    ID : str
        Label for the problem under study.
    beta : float, optional
        Inverse temperature (default: 1e-2)
    n_ensamble : int, optional
        Desired length of the ensamble (default: 1e4)
    n_max : int, optional
        Maximum number of MC moves (default: 4e4)

    Returns
    -------
    P : object
        Current version of P
    Energy : list
        List of the energies for the steps that entered the ensamble
    Ensamble: list
        List of the system states that have been accepted in the ensamble
    elapsed_time : float
        Total elapsed time, in seconds

    '''
    # initialize
    time_start = time.clock()
    P.set_beta(beta)
    Energy = []
    Ensamble = []
    out = open('log_ensamble_%s.dat' % ID, 'w')
    out.write('# start - %s\n' % time.strftime('%c'))
    out.write('# beta: %f\n' % beta)
    out.write('# n_ensamble / n_max: %f / %f\n' %(n_ensamble, n_max))
    out.write('# initial energy: %f\n' % P.energy)
    out.write('#\n')
    out.write('# energy acceptance step\n')
    out.flush()
    # annealing loop
    acc = 0
    count = 0
    while acc <= n_ensamble and count <= n_max:
        accepted = P.MC_move()
        count += 1
        if accepted:
            acc += accepted
            acc_ratio = acc / float(count)
            out.write('%10.4g %.8f %.2g\n' % (P.energy, acc_ratio, P.step))
            out.flush()
            Energy.append(P.energy)
            Ensamble.append(P.state)
        else:
            acc_ratio = acc / float(count)
        P.update_MC_parameters(acc_ratio)
    # finalize
    out.write('# end\n')
    elapsed_time = time.clock() - time_start
    out.write('# elapsed: %.2f s\n' % elapsed_time)
    out.close()
    return P, Energy, Ensamble, elapsed_time
