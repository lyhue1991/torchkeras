import numpy as np
import PIL
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import sys 

simhei_file = str(Path(__file__).parent/"assets/SimHei.ttf")

"""
@author : lyhue1991
@description : keras plot code
"""

def set_matplotlib_font(font_size=15):
    """
    Set the font style and size for Matplotlib plots.

    Args:
        font_size (int): Font size to set.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    try:
        import shutil
        shutil.rmtree(mpl.get_cachedir())
    except:
        pass
    simhei_file = Path(__file__).parent/"assets/SimHei.ttf"
    ttf_dir = Path(mpl.__path__[0])/'mpl-data'/'fonts'/'ttf'
    shutil.copy(str(simhei_file),str(ttf_dir))
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.size'] = str(font_size)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"set matplotlib font to SimHei and fontsize to {font_size}")


def plot_metric(dfhistory, metric):
    """
    Plot training and validation metrics over epochs using Plotly.

    Args:
        dfhistory (pd.DataFrame): DataFrame containing training history.
        metric (str): Name of the metric to plot.

    Returns:
        go.Figure: Plotly figure.
    """
    import plotly.graph_objs as go
    train_metrics = dfhistory["train_"+metric].values.tolist()
    val_metrics = dfhistory['val_'+metric].values.tolist()
    epochs = list(range(1, len(train_metrics) + 1))

    train_scatter = go.Scatter(x=epochs, y=train_metrics, mode="lines+markers",
                              name='train_'+metric, marker=dict(size=8, color="blue"),
                              line=dict(width=2, color="blue", dash="dash"))
    val_scatter = go.Scatter(x=epochs, y=val_metrics, mode="lines+markers",
                            name='val_'+metric, marker=dict(size=10, color="red"),
                            line=dict(width=2, color="red", dash="solid"))
    fig = go.Figure(data=[train_scatter, val_scatter])

    if any([m in metric.lower() for m in ['loss','mse','mae','error','err','l1','l2']]):
        best_metric = np.array(val_metrics).min()
    elif any([m in metric.lower() for m in ['acc','iou','recall','precision','auc']]):
        best_metric = np.array(val_metrics).max()
    elif train_metrics.argmax() > train_metrics.argmin():
        best_metric = np.array(val_metrics).max()
    else:
        best_metric = np.array(val_metrics).min()

    fig.update_layout({"title": "best val_"+metric+"="+f"{best_metric:.4f}",
                       "title_x": 0.45,
                       "xaxis.title": "epoch",
                       "yaxis.title": metric,
                       "font.size": 15,
                       "height": 500,
                       "width": 800
                      })

    return fig


def plot_importance(features, importances, topk=20):
    """
    Plot feature importance using Plotly.

    Args:
        features (list): List of feature names.
        importances (list): List of feature importances.
        topk (int): Number of top features to display.

    Returns:
        go.Figure: Plotly figure.
    """
    import pandas as pd
    import plotly.express as px
    dfimportance = pd.DataFrame({'feature': features, 'importance': importances})
    dfimportance = dfimportance.sort_values(by="importance").iloc[-topk:]
    fig = px.bar(dfimportance, x="importance", y="feature", title="Feature Importance")
    return fig


def plot_score_distribution(labels, scores):
    """
    Plot the score distribution for binary classification problems using Plotly.

    Args:
        labels (list): True labels.
        scores (list): Predicted scores.

    Returns:
        go.Figure: Plotly figure.
    """
    import plotly.express as px
    fig = px.histogram(
        x=scores, color=labels, nbins=50,
        title="Score Distribution",
        labels=dict(color='True Labels', x='Score')
    )
    return fig


def vis_dataframe(df, img_col='img', img_width=100, img_height=100):
    """
    Visualize a DataFrame containing image paths using HTML.

    Args:
        df (pd.DataFrame): DataFrame containing image paths.
        img_col (str): Column name containing image paths.
        img_width (int): Width of displayed images.
        img_height (int): Height of displayed images.

    Returns:
        IPython.display.HTML: HTML object for displaying images.

    """
    from IPython.display import HTML
    dfshow = df.copy()
    dfshow[img_col] = [f'<img src="{x}"  width="{img_width}" height="{img_height}" />'
                       for x in df[img_col]]
    return HTML(dfshow.to_html().replace('&lt;img', '<img').replace('/&gt;', '/>'))


class Colors:
    def __init__(self):
        self.hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in self.hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
    
    def get_hex_color(self,i):
        return self.hexs[int(i)%self.n]

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

class Annotator:
    def __init__(self, img, line_width=None, font_size=None):
        """
        Annotator class for adding bounding boxes, labels, and masks to images.

        Args:
            img (PIL.Image.Image or np.ndarray): Input image.
            line_width (int): Width of the bounding box lines.
            font_size (int): Font size for labels.
        """
        assert np.asarray(img).data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to input images.'
        self.img = img if isinstance(img, Image.Image) else Image.fromarray(img)

        try:
            path = Path(__file__)
            font = str(path.parent/"assets/Arial.ttf")
            size = font_size or max(round(sum(self.img.size) / 2 * 0.035), 12)
            self.font = ImageFont.truetype(font, size)
        except Exception as err:
            print(err, file=sys.stderr)
            self.font = ImageFont.load_default()

        self.lw = line_width or max(round(sum(self.img.size) / 2 * 0.003), 2)

    def add_box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        Add a bounding box with a label to the image.

        Args:
            box (tuple): Coordinates of the bounding box (x1, y1, x2, y2).
            label (str): Label text to display.
            color (tuple): RGB tuple representing the box color.
            txt_color (tuple): RGB tuple representing the label text color.

        Example:
            ```python
            annotator.add_box_label((x1, y1, x2, y2), label='example', color=(128, 128, 128), txt_color=(255, 255, 255))
            ```
        """
        self.draw = ImageDraw.Draw(self.img)

        self.draw.rectangle(box, width=self.lw, outline=color)  # box
        if label:
            
            _, _, w, h = self.font.getbbox(label)  # text width, height 
            outside = box[1] - h >= 0  # label fits outside box
            self.draw.rectangle(
                (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                 box[1] + 1 if outside else box[1] + h + 1),
                fill=color,
            )
            self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)

    def add_mask(self, mask, color, alpha=0.5):
        """
        Add a mask to the image.

        Args:
            mask (np.ndarray): Binary mask array.
            color (tuple): RGB tuple representing the mask color.
            alpha (float): Opacity of the mask.

        Example:
            ```python
            annotator.add_mask(mask, color=(255, 0, 0), alpha=0.5)
            ```
        """
        img = np.asarray(self.img).copy()

        im1_shape = mask.shape
        im0_shape = img.shape

        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2

        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
        mask = mask[top:bottom, left:right]

        mask_img = Image.fromarray((255 * mask).astype(np.uint8), mode='L')
        mask = np.array(mask_img.resize(self.img.size, resample=Image.BILINEAR)) >= 1
        img[mask] = img[mask] * (1 - alpha) + np.array(color) * alpha
        self.img = Image.fromarray(img)


def plot_detection(img, boxes, class_names, min_score=0.2):
    """
    Plot bounding boxes on an image for object detection.

    Args:
        img (PIL.Image.Image or np.ndarray): Input image.
        boxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        class_names (list): List of class names corresponding to the boxes.
        min_score (float): Minimum confidence score to display a box.

    Returns:
        PIL.Image.Image: Annotated image.
    """
    annotator = Annotator(img)
    for *box, conf, cls in reversed(boxes):
        if conf > min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.add_box_label(box, label, color=colors(cls))
    return annotator.img


def plot_instance_segmentation(img, boxes, masks, class_names, min_score=0.2):
    """
    Plot bounding boxes and masks on an image for instance segmentation.

    Args:
        img (PIL.Image.Image or np.ndarray): Input image.
        boxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        masks (list): List of masks corresponding to the boxes.
        class_names (list): List of class names corresponding to the boxes.
        min_score (float): Minimum confidence score to display a box.

    Returns:
        PIL.Image.Image: Annotated image.
    """
    annotator = Annotator(img)
    for (*box, conf, cls), mask in zip(reversed(boxes), reversed(masks)):
        if conf > min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.add_box_label(box, label, color=colors(cls))
            annotator.add_mask(mask.cpu().numpy(), colors(cls))
    return annotator.img


def vis_detection(image, prediction, class_names=None, min_score=0.2, figsize=(16, 16), linewidth=2, color='lawngreen'):
    """
    Visualize object detection results using Matplotlib.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        prediction (dict): Prediction results containing 'boxes', 'scores', and 'labels'.
        class_names (list): List of class names.
        min_score (float): Minimum confidence score to display a box.
        figsize (tuple): Size of the Matplotlib figure.
        linewidth (int): Width of the bounding box lines.
        color (str): Color of the bounding boxes.

    Example:
        ```python
        vis_detection(image, prediction, class_names, min_score=0.5)
        ```
    """
    boxes = np.array(prediction['boxes'].tolist())
    scores = np.array(prediction['scores'].tolist()) if 'scores' in prediction else [1.0 for _ in boxes]
    labels = np.array(prediction['labels'].tolist()) if 'labels' in prediction else None
    classes = [class_names[x] for x in labels] if (
        class_names is not None and labels is not None) else ['object'] * len(boxes)

    import matplotlib.pyplot as plt
    image = np.array(image, dtype=np.uint8)
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()

    for i, (box, name, score) in enumerate(zip(boxes, classes, scores)):
        if score >= min_score:
            text = "{}: {:.2f}".format(name, score)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
            )
            ax.add_patch(patch)
            ax.text(
                x1,
                y1,
                text,
                bbox={"facecolor": color, "alpha": 0.8},
                clip_box=ax.clipbox,
                clip_on=True,
            )
    plt.show();


def joint_imgs_row(img1, img2):
    """
    Concatenate two images horizontally.

    Args:
        img1 (PIL.Image.Image or np.ndarray): First image.
        img2 (PIL.Image.Image or np.ndarray): Second image.

    Returns:
        PIL.Image.Image: Concatenated image.

    """
    size1 = img1.size
    size2 = img2.size
    joint = Image.new('RGB', (size1[0] + size2[0], max(size1[1], size2[1])))
    loc1, loc2 = (0, 0), (size1[0], 0)
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    return joint


def joint_imgs_col(img1, img2):
    """
    Concatenate two images vertically.

    Args:
        img1 (PIL.Image.Image or np.ndarray): First image.
        img2 (PIL.Image.Image or np.ndarray): Second image.

    Returns:
        PIL.Image.Image: Concatenated image.

    """
    size1 = img1.size
    size2 = img2.size
    joint = Image.new('RGB', (max(size1[0], size2[0]), size1[1] + size2[1]))
    loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    return joint


def get_color_map_list(num_classes):
    """
    Get a color map for segmentation masks.

    Args:
        num_classes (int): Number of classes.
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    return color_map


def gray2pseudo(gray_img):
    """
    Convert a grayscale image to a pseudocolored image.

    Args:
        gray_img (Image): Grayscale image.

    Returns:
        Image: Pseudocolored image.
    """
    color_map = get_color_map_list(256)
    gray_p = gray_img.convert('P')
    gray_p.putpalette(color_map)
    return gray_p


def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image.

    Args:
        fig: Matplotlib figure.

    Returns:
        Image: PIL Image.
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def text2img(text):
    """
    Convert text to a PIL Image.

    Args:
        text (str): Text content.

    Returns:
        Image: PIL Image.
    """
    path = Path(__file__)
    simhei = path.parent / "assets/SimHei.ttf"
    lines = len(text.split("\n"))
    image = Image.new("RGB", (800, lines * 20), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(simhei), 18)
    draw.text((0, 0), text, font=font, fill="#000000")
    return image


def img2tensor(image):
    """
    Convert a PIL Image to a PyTorch tensor.

    Args:
        image (Image): PIL Image.

    Returns:
        Tensor: PyTorch tensor.
    """
    from torchvision.transforms import ToTensor
    tensor = ToTensor()(np.array(image))
    return tensor
