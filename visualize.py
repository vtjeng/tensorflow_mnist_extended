import numpy as np
import matplotlib.pyplot as plt
import itertools

def view_images(images, assigned_labels, correct_labels, num_rows, num_columns, fig_id):
    fig = plt.figure(fig_id, figsize=(num_columns+2, num_rows+2))
    for i in xrange(num_rows*num_columns):
        ax = fig.add_subplot(num_rows,num_columns,i+1)
        image = np.reshape(images[i], [28, 28])
        assigned_label = assigned_labels[i]
        correct_label = correct_labels[i]
        plt.title(assigned_label)
        if assigned_label != correct_label:
            ax.title.set_color('red')
        plt.imshow(image, cmap=plt.cm.gray)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
    plt.tight_layout()
    return fig


def view_incorrect(images, assigned_labels, correct_labels, num_rows, num_columns, fig_id):
    mask = (assigned_labels != correct_labels)
    im_sub = list(itertools.compress(images, mask))
    al_sub = list(itertools.compress(assigned_labels, mask))
    cl_sub = list(itertools.compress(correct_labels, mask))
    return view_images(im_sub, al_sub, cl_sub, num_rows, num_columns, fig_id)


def one_hot_to_index(labels):
    return np.argmax(labels, 1)