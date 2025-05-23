"""Show aerosol concentration in room with, without ventilation.

https://rijksoverheid.bouwbesluit.com/Inhoud/docs/wet/bb2012/hfd3/afd3-6
https://rijksoverheid.bouwbesluit.com/Inhoud/docs/wet/bb2012/hfd3/afd3-6?tableid=docs/wet/bb2012[26]/hfd3/afd3-6/par3-6-1

https://www.onlinebouwbesluit.nl/?v=11 (1992)

1 dm³/s = 3.6 m³/h

Ventilatie-eisen Nederlands Bouwbesluit 2012
(Woonkamer: norm verblijfsruimte)
======================================================
Type             Eenheid   Nieuwbouw    Bestaande bouw
======================================================
Kantoor          m³/h/p    23.4         12.4
Onderwijs        m³/h/p    30.6         12.4
Woonkamer 36 m²  m³/h      90           -
   (4 personen)  m³/h/p    22.5         -
======================================================

Bouwbesluit 1992
(geen onderscheid verblijfsgebied/verblijfsruimte)
====================================
Type             Eenheid   Nieuwbouw
====================================
Kantoor          m³/h/m²   4.0
Woonkamer 36 m²  m³/h      116
   (4 personen)  m³/h/p    29
====================================


Created on Tue Dec 14 20:55:29 2021
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock

class Simulation:
    """Time data of concentnrations etc.

    Initialization params:

    - seq: list of (t, Vrate, n)
      (time, volume rate /h, number of persons)
    - nself: number of persons who don't count.
    - desc: title string for plot labels
    - V: room volume
    - tstep: timestep in h.
    - method: 'cn' or 'eu' (for testing only)

    Attributes:

    - tms: time array
    - vrs: volume-rate array (m3/s)
    - ns: number-of-persons array
    - concs: concentrations array (1/m3)
    - concs_s: concentrations array from self
    - concs_v: concentrations from visitors
    - exps: cumulative-exposures array (s/m3)
    - exps_v2s: exposures visitor->self
    - exps_s2v: exposures self->visitors

    Methods:

    - plot()
    """

    def __init__(self, seq, nself=0, desc='Scenario', V=75.0, tstep=1/60,
                 method='cn'):
        """See class doc"""
        # fill arrays
        tstart, tend = seq[0][0], seq[-1][0]
        tms = np.arange(tstart, tend+tstep/2, tstep)
        vrs = np.zeros_like(tms)
        ns = np.zeros_like(tms)
        for t, vr, n in seq:
            ia = int(t/tstep + 0.5)
            vrs[ia:] = vr
            ns[ia:] = n

        ns_self = np.full(ns.shape, nself)
        ns_self = np.where(ns < nself, ns, ns_self)
        ns_vis = ns - ns_self

        # simulate
        self.concs = self._simulate(tms, vrs, ns, V, method=method)
        self.concs_v = self._simulate(tms, vrs, ns_vis, V, method=method)
        self.concs_s = self._simulate(tms, vrs, ns_self, V, method=method)

        self.tms = tms
        self.vrs = vrs
        self.ns = ns

        # cumulative is a bit more tricky. Visitors exposure only counts when
        # they're present.
        self.exps = np.cumsum(self.concs) * tstep
        self.exps_v2s = np.cumsum(self.concs_v) * tstep

        concs_s2v = np.where(ns_vis > 0, self.concs_s, 0)
        self.exps_s2v = np.cumsum(concs_s2v) * tstep
        self.desc = desc

    @staticmethod
    def _simulate(tms, vrs, ns, V, method='cn'):
        """Simulation without accounting for us/them distribution.

        Return: concentrations array.
        """

        # simulate
        concs = np.zeros_like(tms)
        assert method in ('eu', 'cn')

        kdt = np.nan  # depletion per timestep.
        tstep = np.nan
        kdt_max = -1
        for i in range(len(vrs) - 1):
            # Differential equation:
            # dc/dt = -k c + n
            # with k = vr/V
            tstep = tms[i+1] - tms[i]
            kdt = vrs[i]/V * tstep
            kdt_max = max(kdt, kdt_max)
            n = ns[i]
            if method == 'eu':
                # Euler method - not accurate, for testing only
                concs[i+1] = concs[i] * (1 - kdt) + n*tstep/V
            else:
                # Crank-Nicholson implicit method - more accurate
                # even at large time steps
                concs[i+1] = (concs[i] * (1 - kdt/2) + n*tstep/V) / (1 + kdt/2)


        if kdt_max > 0.5:
            print(f'Warning: k*dt={kdt_max:.2g} should be << 1')

        return concs



    def plot(self, *args):
        """Plot data. Optionally also plot others as specified.

        Set split_concs=False to only show total concentrations and not
        separately for 'bewoners' and visitors.
        """

        # TODO: deze inputs in de main() zetten
        split_concs =  st.sidebar.selectbox("Plot seperate lines for inhabitants and visitors", [True, False], index=1)
        danger_line = st.sidebar.number_input("Horizonal line safe CO2 concentration", 0, 10000,900)


        datasets = [self] + list(args)
        # with _lock:
        fig, axs = plt.subplots(4, 1, tight_layout=True, sharex=True,
                                figsize=(7, 9))
        axs[0].set_title('\n\nAantal personen')
        axs[1].set_title('Ventilatiesnelheid (m³/h)')

        if split_concs:
            axs[2].set_title(
                'Aerosolconcentratie (m$^{-3}$)\n'
                'Van bezoek (—), van bewoners (- -), totaal (⋯)')
            axs[3].set_title(
                'Aerosoldosis (h m$^{-3}$)\n'
                'bezoek → bewoners (—), bewoners → bezoek (- -)')
        else:
            axs[2].set_title('Aerosolconcentratie (m$^{-3}$)')
            axs[3].set_title('Aerosoldosis (h m$^{-3}$)')
        axs[3].set_xlabel('Tijd (h)')

        from matplotlib.ticker import MaxNLocator
        axs[3].xaxis.set_major_locator(MaxNLocator(steps=[1,2,5,10]))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 3

        # dash styles with phase difference to reduce overlap.
        dash_styles = [(i, (3, 3)) for i in [0, 1, 2]] * 2
        for i, (ds, col, dstyle) in enumerate(zip(datasets, colors, dash_styles)):

            xargs = dict(zorder=-i, alpha=0.7)
            ax = axs[0]
            ax.plot(ds.tms, ds.ns, **xargs)

            ax = axs[1]
            ax.plot(ds.tms, ds.vrs, color=col, label=ds.desc, **xargs)

            ax = axs[2]
            if split_concs:
                ax.plot(ds.tms, ds.concs_s, color=col, ls='-', **xargs)
                ax.plot(ds.tms, ds.concs_v, color=col, ls=dstyle, **xargs)
                ax.plot(ds.tms, ds.concs, color=col, ls=':', **xargs)
            else:
                ax.plot(ds.tms, ds.concs, color=col, ls='-', **xargs)

            # get equivalent CO2 ppm values
            c2ppm = lambda c: 420 + c*1.67e4
            ppm2c = lambda p: (p-420)/1.67e4

            ax2 = ax.secondary_yaxis("right", functions=(c2ppm, ppm2c))
            ax2.set_ylabel('CO$_2$ conc (ppm)')
            ax.axhline(y=( (danger_line-420)/1.67e4), color="red", alpha=0.6, linestyle="--")


            ax = axs[3]
            # ax.plot(ds.tms, ds.exps, color=col, **xargs)
            if split_concs:
                ax.plot(ds.tms, ds.exps_v2s, color=col, ls='-', **xargs)
                ax.plot(ds.tms, ds.exps_s2v, color=col, ls=dstyle, **xargs)
            else:
                ax.plot(ds.tms, ds.exps, color=col, ls='-', **xargs)

        for ax in axs:
            ax.tick_params(axis='y', which='minor')
            ax.tick_params(axis='both', which='major')
            ax.grid()
            # ax.legend()

        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        for ax in axs:
            ymax = ax.get_ylim()[1]
            ax.set_ylim(-ymax/50, ymax*1.1)
        st.pyplot(fig)


def plot_co2_ventilation():
    #with _lock:
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    vrs = np.linspace(3, 100, 100)
    emission_one = 0.4/24  # m3/h CO2 per person

    def vr2ppm(vr):
        return 420 + emission_one / vr * 1e6

    concs = vr2ppm(vrs)
    ax.plot(vrs, concs)
    ax.set_ylim(0, 4000)
    ax.set_xlim(4, 100)
    ax.set_xscale('log')
    ax.set_xlabel('Ventilatiedebiet per persoon (m$^3$/h)')
    ax.set_ylabel('CO$_2$ concentratie (ppm)')
    xts = [4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ax.set_xticks(xts)
    xtlabs = [
        (str(x) if str(x)[0] in '1234568' else '')
        for x in xts
        ]
    ax.set_xticklabels(xtlabs)
    ax.axhline(420, color='k', ls='--')
    ax.text(4.1, 500, 'Buitenlucht')
    ax.set_title('Aanname: 1 persoon = 17 L/h CO$_2$')

    # Bouwbesluit
    vrs_bb = [
        ('Norm kantoren NL', 23.4),
        ('Norm onderwijs NL', 30.6),
        ('*Advies België', 35), # https://www.info-coronavirus.be/nl/ventilatie/
        ]
    for label, vr in vrs_bb:
        c = vr2ppm(vr)
        ax.scatter([vr], [c], color='k')
        if label.startswith('*'):
            ax.text(vr, c-200, label[1:], ha='right')
        else:
            ax.text(vr, c+70, label)

    # _locator(LogLocator(labelOnlyBase=False))
    # ax.grid(which='minor', axis='x')
    ax.grid()
    st.pyplot(fig)



def main():
    #plt.close('all')
    st.title("Show aerosol concentration in room with, without ventilation.")
    plot_co2_ventilation()

    opp_ = st.sidebar.number_input ("Oppervlakte ruimte (m2)", 0.1, 1000.0, 28.0)

    tstep_ = 0.1 # st.sidebar.number_input("timestep (h)", 0.0, 10.0, 0.1)
    nself_ = st.sidebar.number_input("aantal bewoners", 0, 100, 2)
    gasten_ = st.sidebar.number_input("aantal gasten", 0, 100, 4)
    tot_aanwezigen_ = nself_ + gasten_
    st_airflow  = st.sidebar.number_input("standaard ventilatiesnelh (m³/h)", 0.0, 1000.0, opp_ * 2.5, 50.0)
    st.sidebar.write(f'in bouwbesluit : {opp_} * 2.5 = {opp_* 2.5} m³/h ')
    na_bezoek_airflow  = st.sidebar.number_input(" ventilatiesnelh luchten na bezoek (m³/h)", 0, 1000, 360, 50)
    extra_bezoek_airflow  = st.sidebar.number_input("ventilatiesnelh luchten tijdens bezoek (m³/h)", 0, 1000, 200, 50)
    bezoek_komt  = st.sidebar.number_input("moment dat bezoek komt (h)", 0, 100, 2)
    bezoek_gaat  = st.sidebar.number_input("moment dat bezoek gaat  (h)", 0, 100, 5)

    luchttijd  = st.sidebar.number_input("Luchttijd in scenario 2  (h)", 0.0, 100.0, 0.5)
    eindtijd  = st.sidebar.number_input("Eindtijd in grafiek  (h)", 0, 100, 10)
    hoogte_ = st.sidebar.number_input ("Hoogte ruimte (m)", 0.1, 1000.0, 2.5)
    v_ = opp_ * hoogte_
    st.sidebar.write (f"Room volume = {v_} m3")

    if bezoek_gaat < bezoek_komt:
        st.error("Bezoek kan niet eerder gaan dan komen")
        st.stop()

    simparams = dict(
        V=v_,  # room volume (m3)
        tstep=tstep_,
        method='cn',
        )
    # Note: flow rates tweaked a bit to avoid overlapping lines in plot.
    sim1 = Simulation([
        (0, st_airflow, nself_),
        (bezoek_komt , st_airflow, tot_aanwezigen_),
        (bezoek_gaat, st_airflow, nself_),
        (eindtijd, st_airflow, nself_),
        ],
        nself=nself_, desc='standaard airflow',
        **simparams
        )

    sim2 = Simulation([
        (0, st_airflow+1, nself_),
        (bezoek_komt , st_airflow-1, tot_aanwezigen_),
        (bezoek_gaat, na_bezoek_airflow, nself_),
        (bezoek_gaat+luchttijd, st_airflow+1, nself_),
        (eindtijd, st_airflow+1, nself_),
        ], nself=nself_, desc='Luchten na bezoek',
        **simparams
        )

    sim3 = Simulation([
        (0,  st_airflow-1, nself_),
        (bezoek_komt , extra_bezoek_airflow, tot_aanwezigen_),
        (bezoek_gaat,  st_airflow-1, nself_),
        (eindtijd,  st_airflow-1, nself_),
        ],
        nself=nself_, desc='Extra gedurende bezoek',
        **simparams
        )

    sim1.plot(sim2, sim3)
    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>"
        'Made by <a href="https://twitter.com/hk_nien" target="_blank">Han-Kwang Nienhuys</a><br>'
        'Original : <a href="https://github.com/han-kwang/covid19/blob/master/aerosol_conc_in_room.py" target="_blank">github.com/hk_nien</a><br>'
        'Streamlit-ed by <a href="http://www.twitter.com/rcsmit" target=\"_blank\">@rcsmit</a> <br>'
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/aerosol_in_room_streamlit.py" target="_blank">github.com/rcsmit</a><br>'

        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        '</div>'
    )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)

if __name__ == '__main__':
    main()