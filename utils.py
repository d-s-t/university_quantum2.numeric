from astropy import units
from astropy.units.quantity import Quantity
import astropy.constants as const
import functools
import operator
import plotly.io as pio
from plotly import graph_objects as go

units.MeVc2 = units.def_unit('MeV / c2', units.MeV/const.c**2, format={'latex': r'\mathrm{MeV}/c^2'})

const.nuclee_mass_estimation = 931.49432 * units.MeVc2

const.hbarc = (const.hbar * const.c).to(units.MeV * units.fm)

plotly_show_config = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'unset',
        'width': 800, 'height': 450
    },
    "editable": True,
    # "addButtonToModeBar": [
    #     {
    #         'name': 'save_png',
    #         'title': 'Save as PNG',
    #         'icon': ,
    #         'click': pio.base_renderers.PngRenderer(800, 450, 2).to_mimebundle#lambda fig: pio.write_image(fig, f"./plots/{plotly_show_config['toImageButtonOptions']['filename']}.png", width=800, height=450, format='png', engine='kaleido', scale=2)
    #     }
    # ]
}

# make new template with transparent background
# pio.templates.add(go.layout.Template(name='transparent', layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)')))



def to_latex(q: Quantity, presision=3):
    return q.to_string(format='latex', precision=presision).replace("$", "")

def flatten(a):
    return functools.reduce(operator.iconcat, a, [])


def plotly_export(fig, filename, **kwargs):
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), width=800, height=450)
    fig.write_image(f"./plots/{filename}.svg", width=800, height=450,format='svg', engine='kaleido')
    fig.write_image(f"./plots/{filename}.png", width=800, height=450, format='png', engine='kaleido', scale=2)
    fig.write_image(f"./plots/{filename}.pdf", width=800, height=450, format='pdf', engine='kaleido', scale=2)
    plotly_show_config['toImageButtonOptions']['filename'] = filename
    fig.write_html(f"./plots/{filename}.html", config=plotly_show_config)
    if is_notebook():
        from IPython.display import display, SVG
        display(SVG(filename=f"./plots/{filename}.svg"), **kwargs)
    else:
        fig.show(config=plotly_show_config)
    plotly_show_config['toImageButtonOptions']['filename'] = 'unset'


def progress_bar_range(*n):
    """
    this is generator that uses IPython.display to show a progress bar for the range
    """
    if is_notebook():
        from tqdm.notebook import trange
    else:
        from tqdm import trange
    return trange(*n)
    

def is_notebook():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

def relative_error(E: float, n: int) -> float:
    """
    relative error of the energy
    E: float
        energy in Ry (Rydberg energy)
    n: int
        principal quantum number
    """
    return abs(1 + n**2 * E)

def reduced_mass(*masses):
    """
    reduced mass of a system
    """
    return 1 / sum(1/m for m in masses)


if __name__ == "__main__":
    from time import sleep
    for i in progress_bar_range(10):
        sleep(0.1)

#go.Figure().show(config=plotly_show_config)