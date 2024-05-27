import torch 
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

import PIL
from PIL import Image, ImageDraw ,ImageFont
from matplotlib import pyplot as plt
import torchvision.transforms as T


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = T.Compose(
        [
            T.Resize(target_size),
            T.CenterCrop(target_size),
            T.ToTensor(),
        ]
    )
    image = tform(image)
    image = 2.0 * image - 1.0
    image = image.unsqueeze(0)
    return image


def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")

def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images

def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]    
    return pil_imgs

def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0)*2-1
    elif type(pil_imgs) == list:    
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0)*2-1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs

def add_margin(pil_img, top = 2, right = 2, bottom = 2, 
                    left = 2, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    
    result.paste(pil_img, (left, top))
    return result

def image_grid(imgs, rows = 1, cols = None, 
                    size = None,
                   titles = None, 
                   top=20,
                   font_size = 20, 
                   text_pos = (0, 0), add_margin_size = None):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)
        
    if not size is None:
        imgs = [img.resize((size,size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows*cols
    if not add_margin_size is None:
        imgs = [add_margin(img, top = add_margin_size,
                                right = add_margin_size,
                                bottom = add_margin_size, 
                                left = add_margin_size) for img in imgs]
        
   
    w, h = imgs[0].size
    delta = 0
    if len(imgs)> 1 and not imgs[1].size[1] == h:
        delta = h - imgs[1].size[1] #top
        h = imgs[1].size[1]
    if not titles is  None:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 
                                    size = font_size, encoding="unic")
        h = top + h 
    grid = Image.new('RGB', size=(cols*w, rows*h+delta))    
    for i, img in enumerate(imgs):
        
        if not titles is  None:
            img = add_margin(img, top = top, bottom = 0,left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i],(0,0,0), 
            font = font)
        if not delta == 0 and i > 0:
           grid.paste(img, box=(i%cols*w, i//cols*h+delta))
        else:
            grid.paste(img, box=(i%cols*w, i//cols*h))
        
    return grid    

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#some-more-complex-heatmap-examples
def heatmap(data, row_labels, col_labels, ax=None,
            make_cbar = True, 
            cbar_kw=None, cbarlabel="", **kwargs):
    
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    if make_cbar:
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts