# %% md
"""
## Quantifying n-photon indistinguishability with a cyclic integrated interferometer

https://arxiv.org/pdf/2201.13333.pdf
"""

import numpy as np
from itertools import combinations
from perceval.utils import SVDistribution, StateVector
import perceval as pcvl
from scipy import optimize
import matplotlib.pyplot as plt


def get_inputs_map(sources, circuit):
    inputs_map = None
    for k in range(circuit.m):
        if k in sources:
            distribution = sources[k].probability_distribution()
        else:
            distribution = SVDistribution(StateVector("|0>"))
        # combine distributions
        if inputs_map is None:
            inputs_map = distribution
        else:
            inputs_map *= distribution

    input_StdV = []
    for i in inputs_map.keys():
        input_StdV.append(i)
    for input_state in input_StdV:
        if inputs_map[input_state] == 0:
            del inputs_map[input_state]

    return inputs_map


def outputstate_to_2outcome(output, i):
    """
    :param output: an output of the chip
    :return: a measurable outcome
    """

    if i == 0: channelshom = [0, 1]
    if i == 1: channelshom = [2, 5]
    if i == 2: channelshom = [3, 4]
    if i == 3: channelshom = [6, 7]

    state = []
    for m in output:
        if m.isdigit():
            state.append(m)

    if int(state[channelshom[0]]) == 0 and int(state[channelshom[1]]) == 0:
        return '|0,0>'
    if int(state[channelshom[0]]) == 0 and int(state[channelshom[1]]) > 0:
        return '|0,1>'
    if int(state[channelshom[0]]) > 0 and int(state[channelshom[1]]) == 0:
        return '|1,0>'
    if int(state[channelshom[0]]) > 0 and int(state[channelshom[1]]) > 0:
        return '|1,1>'


def postselect_outputstate(output_state):
    state_list = []
    for m in str(output_state):
        if m.isdigit():
            state_list.append(int(m))

    nb_of_result = 4 - state_list.count(0) + 1

    if nb_of_result < 1:
        return None

    s = []
    for ch, N in enumerate(state_list):
        if N > 0:
            s.append(ch)

    if nb_of_result > 1:
        r = []
        for group in list(combinations(s, r=4)):
            r.append(list(group))
        return r
    else:
        r = [s]
        return r


class Simulator:

    def __init__(self,multiphoton_component: float = 0,
                 M_AB: float = 1, M_BC: float = 1, M_AD: float = 1, M_CD: float = 1,
                 emission_probability: float = 1,
                 losses: float = 0,
                 internal_phase: float = np.pi,
                 nsamples: float = 5e8):

        self.px = emission_probability
        self.losses = losses
        self.g2 = multiphoton_component

        self.phi = internal_phase

        self.nsamples = nsamples

        self.Mij = {'AB': M_AB, 'BC': M_BC, 'AD': M_AD, 'CD': M_CD}
        self.Mi = None

        self.p_fringes = [[1, 3, 5, 7], [1, 2, 3, 6], [0, 3, 5, 6], [0, 2, 3, 7],
                          [1, 4, 5, 6], [1, 2, 4, 7], [0, 4, 5, 7], [0, 2, 4, 6]]
        self.n_fringes = [[1, 3, 5, 6], [1, 2, 3, 7], [0, 3, 5, 7], [0, 2, 3, 6],
                          [1, 4, 5, 7], [1, 2, 4, 6], [0, 4, 5, 6], [0, 2, 4, 7]]

        self.outcome = self.p_fringes + self.n_fringes

        self.circuit = pcvl.Circuit(m=8, name="4PhotonChip")

        for i in range(4):
            self.circuit.add((2 * i, 2 * i + 1), pcvl.BS())

        self.circuit.add(0, pcvl.PS(pcvl.Parameter('alpha')))

        for i in range(3):
            self.circuit.add((2 * i + 1, 2 * i + 2), pcvl.BS())
            self.circuit.add(2 * i + 1, pcvl.PS(phi=pcvl.P(f"phi{i}")))
            self.circuit.add((2 * i + 1, 2 * i + 2), pcvl.BS())

        self.circuit.add((0, 1), pcvl.PERM([1, 0]))
        self.circuit.add((1, 2), pcvl.PERM([1, 0]))
        self.circuit.add((2, 3), pcvl.PERM([1, 0]))

        self.circuit.add((6, 7), pcvl.PERM([1, 0]))
        self.circuit.add((5, 6), pcvl.PERM([1, 0]))
        self.circuit.add((4, 5), pcvl.PERM([1, 0]))

        self.circuit.add((3, 4), pcvl.BS())
        self.circuit.add(3, pcvl.PS(phi=pcvl.P(f"phi{3}")))
        self.circuit.add((3, 4), pcvl.BS())

        self.phase_shifters = self.circuit.get_parameters()

        self.job_corr = None
        self.job_uncorr = None

    def find_Mi(self):

        def get_Vij_from_Vi(V_A, V_B, V_C, V_D):
            # Imperfect single photon source
            source1 = pcvl.Source(multiphoton_component=0,
                                  indistinguishability=V_A,
                                  emission_probability=self.px,
                                  losses=self.losses)
            source2 = pcvl.Source(multiphoton_component=0,
                                  indistinguishability=V_B,
                                  emission_probability=self.px,
                                  losses=self.losses,
                                  context=source1._context, )
            source3 = pcvl.Source(multiphoton_component=0,
                                  indistinguishability=V_C,
                                  emission_probability=self.px,
                                  losses=self.losses,
                                  context=source2._context)
            source4 = pcvl.Source(multiphoton_component=0,
                                  indistinguishability=V_D,
                                  emission_probability=self.px,
                                  losses=self.losses,
                                  context=source3._context)

            mzi = pcvl.Circuit(2) // pcvl.BS() // (1, pcvl.PS(phi=pcvl.P("phi"))) // pcvl.BS()

            # Compute the output distribution for phi2=np.pi/2
            phase = mzi.get_parameters()[0]
            phase.set_value(np.pi / 2)

            outcome = {"AB": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       "BC": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       "AD": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       "CD": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       }

            input_state = {"AB": get_inputs_map({0: source1, 1: source2}, mzi),
                           "BC": get_inputs_map({0: source2, 1: source3}, mzi),
                           "AD": get_inputs_map({0: source1, 1: source4}, mzi),
                           "CD": get_inputs_map({0: source3, 1: source4}, mzi),
                           }

            p = {"AB": pcvl.Processor("SLOS", mzi, input_state['AB']),
                 "BC": pcvl.Processor("SLOS", mzi, input_state['BC']),
                 "AD": pcvl.Processor("SLOS", mzi, input_state['AD']),
                 "CD": pcvl.Processor("SLOS", mzi, input_state['CD']),
                 }

            processed = {}
            for party in p:
                p[party].with_input(input_state[party])
                p[party].mode_post_selection(1)
                sampler_uncorr = pcvl.algorithm.Sampler(p[party])
                job = sampler_uncorr.sample_count(self.nsamples)

                processed[party] = job['results']

            for party in processed:

                probability_distribution = processed[party]

                for output_state in probability_distribution:
                    # print(str(output_state), str(sv_out[output_state]))
                    # Each output is mapped to an outcome
                    result = outputstate_to_2outcome(str(output_state), 0)
                    # The probability of an outcome is added, weighted by the probability of this input
                    outcome[party][result] += probability_distribution[output_state]

            p_corr = {"AB": outcome["AB"]['|1,1>'],
                      "BC": outcome["BC"]['|1,1>'],
                      "AD": outcome["AD"]['|1,1>'],
                      "CD": outcome["CD"]['|1,1>']
                      }

            # Compute the output distribution for phi2=np.pi/2
            phase = mzi.get_parameters()[0]
            phase.set_value(0)

            outcome = {"AB": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       "BC": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       "AD": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       "CD": {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       }

            p = {"AB": pcvl.Processor("SLOS", mzi, input_state['AB']),
                 "BC": pcvl.Processor("SLOS", mzi, input_state['BC']),
                 "AD": pcvl.Processor("SLOS", mzi, input_state['AD']),
                 "CD": pcvl.Processor("SLOS", mzi, input_state['CD']),
                 }

            processed = {}
            for party in p:
                p[party].with_input(input_state[party])
                p[party].mode_post_selection(1)
                sampler_uncorr = pcvl.algorithm.Sampler(p[party])
                job = sampler_uncorr.sample_count(self.nsamples)

                processed[party] = job['results']

            for party in processed:

                probability_distribution = processed[party]

                for output_state in probability_distribution:
                    # print(str(output_state), str(sv_out[output_state]))
                    # Each output is mapped to an outcome
                    result = outputstate_to_2outcome(str(output_state), 0)
                    # The probability of an outcome is added, weighted by the probability of this input
                    outcome[party][result] += probability_distribution[output_state]

            p_uncorr = {"AB": outcome["AB"]['|1,1>'],
                        "BC": outcome["BC"]['|1,1>'],
                        "AD": outcome["AD"]['|1,1>'],
                        "CD": outcome["CD"]['|1,1>'],
                        }

            V_AB = 1 - 2 * p_corr["AB"] / p_uncorr["AB"]
            V_BC = 1 - 2 * p_corr["BC"] / p_uncorr["BC"]
            V_AD = 1 - 2 * p_corr["AD"] / p_uncorr["AD"]
            V_CD = 1 - 2 * p_corr["CD"] / p_uncorr["CD"]

            return V_AB, V_BC, V_AD, V_CD

        def residual(x, Mexp):
            M_ABexp, M_BCexp, M_ADexp, M_CDexp = Mexp

            V_A, V_B, V_C, V_D = x

            V_AB, V_BC, V_AD, V_CD = get_Vij_from_Vi(V_A, V_B, V_C, V_D)

            error = np.sqrt(
                (V_AB - M_ABexp) ** 2 + (V_BC - M_BCexp) ** 2 + (V_AD - M_ADexp) ** 2 + (V_CD - M_CDexp) ** 2
            )

            return error

        Mij_exp = [self.Mij['AB'], self.Mij['BC'], self.Mij['AD'], self.Mij['CD']]

        # Starting point
        init = Mij_exp
        result = optimize.minimize(residual, init, args=Mij_exp, bounds=4 * [(0, 1)])
        Vi = result.x

        self.Mi = {'A': Vi[0], 'B': Vi[1], 'C': Vi[2], 'D': Vi[3]}

    def compute_local(self, mzi_settings='correlated'):

        assert self.Mi is not None, 'You must first find_Mi()'

        self.phase_shifters[0].set_value(self.phi)

        if mzi_settings == 'correlated':
            for i in range(1, 5):
                self.phase_shifters[i].set_value(np.pi / 2)
        else:
            for i in range(1, 5):
                self.phase_shifters[i].set_value(0)

        # Imperfect single photon source
        source1 = pcvl.Source(multiphoton_component=self.g2,
                              indistinguishability=self.Mi['A'],
                              emission_probability=self.px,
                              losses=self.losses)
        source2 = pcvl.Source(multiphoton_component=self.g2,
                              indistinguishability=self.Mi['B'],
                              emission_probability=self.px,
                              losses=self.losses,
                              context=source1._context)
        source3 = pcvl.Source(multiphoton_component=self.g2,
                              indistinguishability=self.Mi['C'],
                              emission_probability=self.px,
                              losses=self.losses,
                              context=source2._context)
        source4 = pcvl.Source(multiphoton_component=self.g2,
                              indistinguishability=self.Mi['D'],
                              emission_probability=self.px,
                              losses=self.losses,
                              context=source3._context)

        sources = {0: source1, 2: source2, 4: source3, 6: source4}
        inputs_map = get_inputs_map(sources, self.circuit)

        local_qpu = pcvl.Processor("SLOS", self.circuit, inputs_map)
        local_qpu.mode_post_selection(1)
        local_qpu.with_input(inputs_map)

        sampler = pcvl.algorithm.Sampler(local_qpu)
        job = sampler.sample_count(self.nsamples)

        if mzi_settings == 'correlated':
            self.job_corr = job
        else:
            self.job_uncorr = job

        return job

    def get_results_local(self):

        assert self.job_corr is not None

        empirical_table = {self.phi: {'p': 0, 'n': 0}}

        sv_out = self.job_corr['results']

        for output_state in sv_out:
            result = postselect_outputstate(output_state)
            if result:
                for r in result:
                    if r in self.p_fringes:
                        empirical_table[self.phi]['p'] += sv_out[output_state]
                    if r in self.n_fringes:
                        empirical_table[self.phi]['n'] += sv_out[output_state]

        norm_factor = empirical_table[self.phi]['n'] + empirical_table[self.phi]['p']

        empirical_table[self.phi]['p'] = empirical_table[self.phi]['p'] / norm_factor
        empirical_table[self.phi]['n'] = empirical_table[self.phi]['n'] / norm_factor

        return empirical_table

    def get_results_local_4hom(self):

        assert self.job_corr is not None
        assert self.job_uncorr is not None

        all_outcome = [{'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0}]

        sv_out = self.job_uncorr['results']
        for output_state in sv_out:
            for i in range(4):
                result = outputstate_to_2outcome(str(output_state), i)
                all_outcome[i][result] += sv_out[output_state]
        all_p_uncorr = []
        for outcome in all_outcome:
            all_p_uncorr.append(outcome['|1,1>'])

        all_outcome = [{'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0},
                       {'|0,0>': 0, '|1,0>': 0, '|1,1>': 0, '|0,1>': 0}]

        sv_out = self.job_corr['results']
        for output_state in sv_out:
            for i in range(4):
                result = outputstate_to_2outcome(str(output_state), i)
                all_outcome[i][result] += sv_out[output_state]
        all_p_corr = []
        for outcome in all_outcome:
            all_p_corr.append(outcome['|1,1>'])

        all_V = []
        for qubit in range(4):
            all_V.append(1 - 2 * all_p_corr[qubit] / all_p_uncorr[qubit])

        return all_V

    def plot_histogram(self):
        probability_distribution = self.job_corr['results']

        fig, ax = plt.subplots(constrained_layout=True)
        histo_p = {}
        histo_n = {}
        groups_labels_p = [str([i + 1 for i in g]) for g in self.p_fringes]
        groups_labels_n = [str([i + 1 for i in g]) for g in self.n_fringes]
        for output_state in probability_distribution:
            result = postselect_outputstate(output_state)
            if result:
                for r in result:
                    if r in self.p_fringes:
                        histo_p[output_state] = probability_distribution[output_state]
                    if r in self.n_fringes:
                        histo_n[output_state] = probability_distribution[output_state]
        histo_p_str = {str(k): histo_p[k] for k in histo_p}
        histo_n_str = {str(k): histo_n[k] for k in histo_n}
        ax.bar(histo_p_str.keys(), histo_p_str.values(), yerr=[np.sqrt(i) for i in histo_p_str.values()],
               capsize=4, label='Constructive outputs')
        ax.bar(histo_n_str.keys(), histo_n_str.values(), yerr=[np.sqrt(i) for i in histo_n_str.values()],
               capsize=4, label='Destructive outputs')
        ax.set_xlabel('Output state', fontsize=30)
        ax.set_ylabel('4-photon coincidences', fontsize=30)
        xticklabels = groups_labels_p + groups_labels_n
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=20)
        ax.tick_params(direction='in', bottom=True, top=False, left=True, right=True, labelsize=28)
        plt.show()


if __name__ == '__main__':
    sim = Simulator(multiphoton_component=0.0,
                    M_AB=0.864, M_BC=0.944, M_AD=0.947, M_CD=0.867,
                    emission_probability=0.75,
                    losses=0.97,
                    internal_phase=2 * np.pi,
                    nsamples=1e10)

    sim.find_Mi()
    print("Found the input parameters Mi from the measured overlaps Mij:")
    print(sim.Mi)

    sim.compute_local(mzi_settings='correlated')
    sim.compute_local(mzi_settings='uncorrelated')

    all_V = sim.get_results_local_4hom()
    e_t = sim.get_results_local()

    print('Pairwise 2-photon indistinguishability:')
    print(all_V)

    print('4-photon indistinguishability parameter:')
    print((e_t[2 * np.pi]['n'] - e_t[2 * np.pi]['p']) / (e_t[2 * np.pi]['p'] + e_t[2 * np.pi]['n']))

    sim.plot_histogram()
