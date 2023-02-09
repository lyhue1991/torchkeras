import numpy as np
import PIL
from PIL import Image,ImageDraw,ImageFont
import pathlib
import sys 

def plot_metric(dfhistory, metric):
    import plotly.graph_objs as go
    train_metrics = dfhistory["train_"+metric].values.tolist()
    val_metrics = dfhistory['val_'+metric].values.tolist()
    epochs = list(range(1, len(train_metrics) + 1))
    
    train_scatter = go.Scatter(x = epochs, y=train_metrics, mode = "lines+markers",
                               name = 'train_'+metric,marker = dict(size=8,color="blue"),
                                line= dict(width=2,color="blue",dash="dash"))
    val_scatter = go.Scatter(x = epochs, y=val_metrics, mode = "lines+markers",
                            name = 'val_'+metric,marker = dict(size=10,color="red"),
                            line= dict(width=2,color="red",dash="solid"))
    fig = go.Figure(data = [train_scatter,val_scatter])
    return fig  

def plot_importance(features, importances, topk=20):
    import pandas as pd 
    import plotly.express as px 
    dfimportance = pd.DataFrame({'feature':features,'importance':importances})
    dfimportance = dfimportance.sort_values(by = "importance").iloc[-topk:]
    fig = px.bar(dfimportance,x="importance",y="feature",title="Feature Importance")
    return fig
 
def plot_score_distribution(labels,scores):
    'for binary classification problem.'
    import plotly.express as px 
    fig = px.histogram(
        x=scores, color = labels,  nbins=50,
        title = "Score Distribution",
        labels=dict(color='True Labels', x='Score')
    )
    return fig


class _Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = _Colors()

class Annotator:
    def __init__(self, img, line_width=None, 
                 font_size=None):
        assert np.asarray(img).data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to input images.'
        self.img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        
        try:
            path = pathlib.Path(__file__)
            font = str(path.parent/"assets/Arial.ttf")
            size = font_size or max(round(sum(self.img.size) / 2 * 0.035), 12)
            self.font = ImageFont.truetype(font, size)
        except Exception as err:
            print(err, file = sys.stderr)
            self.font = ImageFont.load_default()
            
        self.lw = line_width or max(round(sum(self.img.size) / 2 * 0.003), 2)  
        self.pil_9_2_0_check = True if  PIL.__version__>='9.2.0' else False

    def add_box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        self.draw = ImageDraw.Draw(self.img)
        
        self.draw.rectangle(box, width=self.lw, outline=color)  # box
        if label:
            if self.pil_9_2_0_check:
                _, _, w, h = self.font.getbbox(label)  # text width, height (New)
            else:
                w, h = self.font.getsize(label)  # text width, height (Old, deprecated in 9.2.0)
            outside = box[1] - h >= 0  # label fits outside box
            self.draw.rectangle(
                (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                 box[1] + 1 if outside else box[1] + h + 1),
                fill=color,
            )
            # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
            self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)


    def add_mask(self, mask,  color, alpha=0.5):
        # Add one mask to image
        img = np.asarray(self.img).copy()
                
        mask_img = Image.fromarray((255*mask).astype(np.uint8),mode='L')
        mask = np.array(mask_img.resize(self.img.size))>=1
        
        img[mask] = img[mask] * (1-alpha) + np.array(color) * alpha
        self.img = Image.fromarray(img)
          
def plot_detection(img, boxes, class_names, min_score=0.2):
    annotator = Annotator(img)
    for *box, conf, cls in reversed(boxes):
        if conf>min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.add_box_label(box, label, color=colors(cls))
    return annotator.img 

def plot_instance_segmentation(img, boxes, masks, class_names, min_score=0.2):
    annotator = Annotator(img)
    for (*box, conf, cls),mask in zip(reversed(boxes),reversed(masks)):
        if conf>min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            annotator.add_box_label(box, label, color=colors(cls))
            annotator.add_mask(mask.cpu().numpy(),colors(cls))
    return annotator.img 
