# some useful customized functions!
from IPython import display

def user_svg_display():
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize = (3.5, 2.5)):
    user_svg_display()
    plt.rcParams['figure.figsize'] = figsize