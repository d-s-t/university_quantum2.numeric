from utils import const, plotly_show_config, to_latex
from preset import H1, H2, He4, K
import numpy as np
from sys import float_info
from numerov_model import Numerov


def q3_1_2():
    v = lambda r: (- const.alpha * H1.Z * const.hbarc / r).to('MeV')
    m = Numerov(H1, v, n=1)
    N = int(1e4)
    r = np.linspace(float_info.epsilon , 10, N)
    E = np.array([-1.1, -1.05, -1, -0.95, -0.9])
    u = m.ure(r, E)
    
    from plotly import graph_objects as go
    fig = go.Figure()
    for i in range(len(E)):
        fig.add_trace(go.Scatter(x=r, y=u[:,i], mode='lines', name=f'E={E[i]}'))
    fig.update_layout(title=r'$ \text{Q3.1.2 - } {}^{1}\!H \text{ base state } \left(n=1,l=0\right) $') \
        .update_xaxes(title=r'$ r \; \left[a_{B} = '+ to_latex(m.a_B) + r'\right] $') \
        .update_yaxes(title=r'$ u\left(r\right) \; \left[fm\right] $', range=[-100, 100])
    # fig.write_image("/plots/q3.1.2.svg")
    plotly_show_config['toImageButtonOptions']['filename'] = 'q3.1.2'
    fig.show(config=plotly_show_config)
    plotly_show_config['toImageButtonOptions']['filename'] = 'unset'


def q3_1_4():
    v = lambda r: (- const.alpha * H1.Z * const.hbarc / r).to('MeV')
    m = Numerov(H1, v, n=1)
    Ns = np.array([int(1e2), int(1e3), int(1e4), int(1e5)])
    etta = np.zeros(Ns.shape)
    for i, N in enumerate(Ns):
        r = np.linspace(float_info.epsilon, 20, N)
        E = m.find_root(-1.05, -0.95, r)[0]
        etta[i] = m.etta(E)

    from plotly import graph_objects as go
    fig = go.Figure() \
        .add_trace(go.Scatter(x=Ns, y=etta, mode='lines+markers')) \
        .update_layout(title=r'$ \text{Q3.1.4 - } {}^{1}\!H \text{ base state } \left(n=1,l=0\right) $') \
        .update_xaxes(type='log', title=r'$ N $') \
        .update_yaxes(type='log', title=r'$ \eta $')
    plotly_show_config['toImageButtonOptions']['filename'] = 'q3.1.4'
    fig.show(config=plotly_show_config)
    plotly_show_config['toImageButtonOptions']['filename'] = 'unset'


if __name__ == '__main__':
    # q3_1_2()
    q3_1_4()