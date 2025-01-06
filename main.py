from constants import const
from preset import H1, H2, He4, K
import numpy as np
from sys import float_info
from numerov_model import Numerov


def q3_1_2():
    v = lambda r: (- const.alpha * H1.Z * const.hbarc / r).to('MeV')
    m = Numerov(H1, v, n=1, l=0)
    N = int(1e4)
    r = np.linspace(float_info.epsilon , 10, N)
    E = np.array([-1.1, -1.05, -1, -0.95, -0.9])
    u = m.ure(r, E)
    
    from plotly import graph_objects as go
    fig = go.Figure()
    for i in range(len(E)):
        fig.add_trace(go.Scatter(x=r, y=u[:,i], mode='lines', name=f'E={E[i]}'))
    fig.update_layout(title=r'$ \text{Q3.1.2 - } {}^{1}\!H \text{ base state } \left(n=1,l=0\right) $',
                      xaxis_title=r'$ r \left[fm\right] $',
                      yaxis=dict(title=r'$ u\left(r\right) \left[fm\right] $', range=[-100, 100]))
    # fig.write_image("/plots/q3.1.2.svg")
    fig.show()


def q3_1_4():
    v = lambda r: (- const.alpha * H1.Z * const.hbarc / r).to('MeV')
    m = Numerov(H1, v, n=1, l=0)
    r = np.linspace(float_info.epsilon, 10, int(1e4))
    E = m.find_root(-1., -0.98, r)
    print(E)
    u = m.ure(r, E)

    from plotly import graph_objects as go
    fig = go.Figure()
    for i in range(len(E)):
        fig.add_trace(go.Scatter(x=r, y=u[:,i], mode='lines', name=f'E={E[i]}'))
    fig.update_layout(title=r'$ \text{Q3.1.4 - } {}^{1}\!H \text{ base state } \left(n=1,l=0\right) $',
                      xaxis_title=r'$ r \left[fm\right] $',
                      yaxis=dict(title=r'$ u\left(r\right) \left[fm\right] $'
                                #  range=[-100, 100],
                                #    type='log'
                                   ))
    # fig.write_image(r"\plots\q3.1.4.png")
    fig.show()


if __name__ == '__main__':
    #q3_1_2()
    q3_1_4()