import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualise(imagelist):
    fig = plt.figure() # make figure

    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(imagelist[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        # return the artists set
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=len(imagelist), 
                                interval=60, blit=True)
    plt.show(block=True)