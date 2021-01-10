# some useful customized functions!
from IPython import display

def user_svg_display():
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize = (3.5, 2.5)):
    user_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
# Optimizer
# Change directly in the same memory
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size