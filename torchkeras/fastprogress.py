'''
This code is borrowed from https://github.com/fastai/fastprogress/
'''
import time,os,shutil
from sys import stdout
from warnings import warn
from .utils import is_jupyter

if is_jupyter():
    try:
        from IPython.display import clear_output, display, HTML
        import matplotlib.pyplot as plt
    except:
        warn("Couldn't import ipywidgets properly, progress bar will use console behavior")
             
def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h,m,s = t//3600, (t//60)%60, t%60
    if h!= 0: return f'{h}:{m:02d}:{s:02d}'
    else:     return f'{m:02d}:{s:02d}'

html_progress_bar_styles = """
<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>
"""

def html_progress_bar(value, total, label, interrupted=False):
    "Html code for a progress bar `value`/`total` with `label`"
    bar_style = 'progress-bar-interrupted' if interrupted else ''
    val = '' if total is None else f"value='{value}'"
    return f"""
    <div>
      <progress {val} class='{bar_style}' max='{total}' style='width:300px; height:20px; vertical-align: middle;'></progress>
      {label}
    </div>
    """

def text2html_table(items):
    "Put the texts in `items` in an HTML table."
    html_code = f"""<table border="1" class="dataframe">\n"""
    html_code += f"""  <thead>\n    <tr style="text-align: left;">\n"""
    for i in items[0]: html_code += f"      <th>{i}</th>\n"
    html_code += f"    </tr>\n  </thead>\n  <tbody>\n"
    for line in items[1:]:
        html_code += "    <tr>\n"
        for i in line: html_code += f"      <td>{i}</td>\n"
        html_code += "    </tr>\n"
    html_code += "  </tbody>\n</table><p>"
    return html_code



class ProgressBar():
    update_every,first_its,lt = 0.2,5,'<'

    def __init__(self, gen, total=None, display=True, leave=True, parent=None, master=None, comment=''):
        self.gen,self.parent,self.master,self.comment = gen,parent,master,comment
        self.total = None if total=='noinfer' else len(gen) if total is None else total
        self.last_v = 0
        if parent is None: self.leave,self.display = leave,display
        else:
            self.leave,self.display=False,False
            parent.add_child(self)
        self.last_v = None

    def on_iter_begin(self):
        if self.master is not None: self.master.on_iter_begin()

    def on_interrupt(self):
        if self.master is not None: self.master.on_interrupt()

    def on_iter_end(self):
        if self.master is not None: self.master.on_iter_end()

    def on_update(self, val, text): pass

    def __iter__(self):
        if self.total != 0: self.update(0)
        try:
            for i,o in enumerate(self.gen):
                if self.total and i >= self.total: break
                yield o
                self.update(i+1)
            if self.total is None and self.last_v is not None:
                self.total = i+1
                self.update(self.total)
        except Exception as e:
            self.on_interrupt()
            raise e

    def update(self, val):
        if self.last_v is None:
            self.on_iter_begin()
            self.last_v = 0
        if val == 0:
            self.start_t = self.last_t = time.time()
            self.pred_t,self.last_v,self.wait_for = None,0,1
            self.update_bar(0)
        elif val <= self.first_its or val >= self.last_v + self.wait_for or (self.total and val >= self.total):
            cur_t = time.time()
            avg_t = (cur_t - self.start_t) / val
            self.wait_for = max(int(self.update_every / (avg_t+1e-8)),1)
            self.pred_t = None if self.total is None else avg_t * self.total
            self.last_v,self.last_t = val,cur_t
            self.update_bar(val)
            if self.total is not None and val >= self.total:
                self.on_iter_end()
                self.last_v = None

    def update_bar(self, val):
        if self.total == 0:
            warn("Your generator is empty.")
            return self.on_update(0, '100% [0/0]')
        if val ==0:
            return self.on_update(0, f'0% [0/{self.total}]')
        pct = '' if self.total is None else f'{100 * val/self.total:.2f}% '
        tot = '?' if self.total is None else str(self.total)
        elapsed_t = self.last_t - self.start_t
        remaining_t = '?' if self.pred_t is None else format_time(self.pred_t - elapsed_t)
        elapsed_t = format_time(elapsed_t)
        end = '' if len(self.comment) == 0 else f' {self.comment}'
        self.on_update(val, f'{pct}[{val}/{tot} {elapsed_t}{self.lt}{remaining_t}{end}]')
        
        
class MasterBar(ProgressBar):
    def __init__(self, gen, cls, total=None):
        self.main_bar = cls(gen, total=total, display=False, master=self)

    def on_iter_begin(self): pass
    def on_interrupt(self):  pass
    def on_iter_end(self):   pass
    def add_child(self, child): pass
    def write(self, line):      pass
    def update_graph(self, graphs, x_bounds, y_bounds): pass

    def __iter__(self):
        for o in self.main_bar:
            yield o

    def update(self, val): self.main_bar.update(val)


class NBProgressBar(ProgressBar):
    lt = '&lt;'
    def on_iter_begin(self):
        super().on_iter_begin()
        self.progress = html_progress_bar(0, self.total, "")
        if self.display:
            display(HTML(html_progress_bar_styles))
            self.out = display(HTML(self.progress), display_id=True)
        self.is_active=True

    def on_interrupt(self):
        self.on_update(0, 'Interrupted', interrupted=True)
        super().on_interrupt()
        self.on_iter_end()

    def on_iter_end(self):
        if not self.leave and self.display: self.out.update(HTML(''))
        self.is_active=False
        super().on_iter_end()

    def on_update(self, val, text, interrupted=False):
        self.progress = html_progress_bar(val, self.total, text, interrupted)
        if self.display: self.out.update(HTML(self.progress))
        elif self.parent is not None: self.parent.show()

class NBMasterBar(MasterBar):
    names = ['train', 'valid']
    def __init__(self, gen, total=None, hide_graph=False, order=None, clean_on_interrupt=False, total_time=False):
        super().__init__(gen, NBProgressBar, total)
        if order is None: order = ['pb1', 'text', 'pb2']
        self.hide_graph,self.order = hide_graph,order
        self.report,self.clean_on_interrupt,self.total_time = [],clean_on_interrupt,total_time
        self.inner_dict = {'pb1':self.main_bar, 'text':""}
        self.text,self.lines = "",[]

    def on_iter_begin(self):
        self.html_code = '\n'.join([html_progress_bar(0, self.main_bar.total, ""), ""])
        display(HTML(html_progress_bar_styles))
        self.out = display(HTML(self.html_code), display_id=True)

    def on_interrupt(self):
        if self.clean_on_interrupt: self.out.update(HTML(''))

    def on_iter_end(self):
        if hasattr(self, 'graph_fig'):
            plt.close()
            self.graph_out.update(self.graph_fig)
        if self.text.endswith('<p>'): self.text = self.text[:-3]
        if self.total_time:
            total_time = format_time(time.time() - self.main_bar.start_t)
            self.text = f'Total time: {total_time} <p>' + self.text
        if hasattr(self, 'out'): self.out.update(HTML(self.text))

    def add_child(self, child):
        self.child = child
        self.inner_dict['pb2'] = self.child

    def show(self):
        self.inner_dict['text'] = self.text
        to_show = [name for name in self.order if name in self.inner_dict.keys()]
        self.html_code = '\n'.join([getattr(self.inner_dict[n], 'progress', self.inner_dict[n]) for n in to_show])
        self.out.update(HTML(self.html_code))

    def write(self, line, table=False):
        if not table: self.text += line + "<p>"
        else:
            self.lines.append(line)
            self.text = text2html_table(self.lines)
        self.show()

    def update_graph(self, dfhistory, metric, x_bounds=None, y_bounds=None, title = None, figsize=(6,4)):
        if self.hide_graph: return
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=figsize)
            self.graph_out = display(self.graph_ax.figure, display_id=True)
            
        self.graph_ax.clear()
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []
        
        m1 = "train_"+metric
        if  m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.graph_ax.plot(epochs,train_metrics,'bo--',label= m1)
        
        m2 = 'val_'+metric
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.graph_ax.plot(epochs,val_metrics,'ro-',label =m2)
            
        if metric in dfhistory.columns:
            metric_values = dfhistory[metric]
            self.graph_ax.plot(epochs, metric_values,'go-', label = metric)
            
        self.graph_ax.set_xlabel("epoch")
        self.graph_ax.set_ylabel(metric)  
        
        if title:
             self.graph_ax.set_title(title)
        
        if m1 in dfhistory.columns or m2 in dfhistory.columns or metric in dfhistory.columns:
            self.graph_ax.legend(loc='best')
        
        if x_bounds is not None: self.graph_ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.graph_ax.set_ylim(*y_bounds)
        self.graph_out.update(self.graph_ax.figure);