'''
reference from https://github.com/fastai/fastprogress/
'''
import time,sys
from IPython.display import clear_output, display, HTML
from tqdm.utils import _term_move_up

"""
@author : lyhue1991
@description : pbar
"""

move_up = _term_move_up()

def is_jupyter():
    """
    Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in Jupyter notebook, False otherwise.
    """
    import contextlib
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False


def format_time(t):
    """
    Format time in seconds to (h):mm:ss.

    Args:
        t (int): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    if h != 0:
        return f'{h}:{m:02d}:{s:02d}'
    else:
        return f'{m:02d}:{s:02d}'


def format_number(x):
    """
    Format a number in scientific notation if too small or too large.

    Args:
        x (float): Number to be formatted.

    Returns:
        str: Formatted number string.
    """
    if abs(x) < 1e-4 or abs(x) >= 1e5:
        return "{:.4e}".format(x)
    else:
        return "{:.4f}".format(x)


# HTML styles for progress bar
html_progress_bar_styles = """
<style>
    /* background: */
    progress::-webkit-progress-bar {background-color: #CDCDCD; width: 100%;}
    progress {background-color: #CDCDCD;}

    /* value: */
    progress::-webkit-progress-value {background-color: #00BFFF  !important;}
    progress::-moz-progress-bar {background-color: #00BFFF  !important;}
    progress {color: #00BFFF ;}

    /* optional */
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #000000;
    }
</style>
"""

def html_progress_bar(value, total, label='', postfix='', interrupted=False):
    """
    Generate HTML code for a progress bar.

    Args:
        value (float): Current progress value.
        total (float): Total progress value.
        label (str): Additional label for the progress bar.
        postfix (str): Additional information to display after the progress bar.
        interrupted (bool): Whether the progress bar is interrupted.

    Returns:
        str: HTML code for the progress bar.
    """
    bar_style = 'progress-bar-interrupted' if interrupted else ''
    val = '' if total is None else f"value='{value}'"
    return f"""
    <div>
      <progress {val} class='{bar_style}' max='{total}' style='width:300px; height:20px; vertical-align: middle;'></progress>
      {label}
      <br>
      {postfix}
    </div>
    """


def text_progress_bar(value, total, label='', postfix='', interrupted=False):
    """
    Generate a text-based progress bar.

    Args:
        value (float): Current progress value.
        total (float): Total progress value.
        label (str): Additional label for the progress bar.
        postfix (str): Additional information to display after the progress bar.
        interrupted (bool): Whether the progress bar is interrupted.

    Returns:
        str: Text-based progress bar string.
    """
    bar_style = "🟦" if interrupted else "⬜️"  # Google style: "🟥", "⬜️" "○", "*"'🟦''⬛️'

    # Calculate the percentage of completion
    percentage = round(value / total * 20)

    # Define the finished and unfinished parts of the progress bar
    finished = "🟩" * percentage  # Google style: "🟩", "●"
    unfinished = bar_style * (20 - percentage)

    # Construct the progress bar string
    bar = "\r{}{} {}".format(finished, unfinished, label) + " " * 20 + "\t" * 50 + f"{postfix}" + " " * 20 + "\t" * 50
    return bar

class ProgressBar:
    update_every, first_its, lt = 0.2, 5, '<'

    def __init__(self, gen, total=None, comment='', comment_tail=''):
        """
        Initialize the ProgressBar instance.

        Args:
            gen: Iterable to track progress.
            total: Total number of iterations (None for unknown).
            comment: Initial comment to display.
            comment_tail: Additional tail for the comment.
        """
        # Initialize instance variables
        self.gen, self.comment, self.comment_tail = gen, comment, comment_tail
        self.postfix = ''
        self.total = None if total == 'noinfer' else len(gen) if total is None else total
        self.last_v = None
        self.display = True
        self.in_jupyter = is_jupyter()
        self.update(0)

    def update(self, val):
        """
        Update the progress bar.

        Args:
            val: Current value of the progress.
        """
        # Initialization on the first iteration
        if self.last_v is None:
            self.on_iter_begin()
            self.last_v = 0

        # Initial update or update triggered by specific conditions
        if val == 0:
            self.start_t = self.last_t = time.time()
            self.pred_t, self.last_v, self.wait_for = None, 0, 1
            self.update_bar(0)
        elif val <= self.first_its or val >= self.last_v + self.wait_for or (self.total and val >= self.total):
            cur_t = time.time()
            avg_t = (cur_t - self.start_t) / val
            self.wait_for = max(int(self.update_every / (avg_t + 1e-8)), 1)
            self.pred_t = None if self.total is None else avg_t * self.total
            self.last_v, self.last_t = val, cur_t

            # Check if the last iteration is reached
            if self.total is not None and val >= self.total:
                self.on_iter_end()
                self.last_v = self.total
            else:
                self.update_bar(val)

    def on_iter_begin(self):
        """
        Initialize progress bar at the beginning of an iteration.
        """
        if self.in_jupyter:
            # Display progress bar in Jupyter environment
            self.html_code = '\n'.join([html_progress_bar(0, self.total, ""), ""])
            display(HTML(html_progress_bar_styles))
            self.out = display(HTML(self.html_code), display_id=True)
        else:
            print('\n')

    def on_iter_end(self):
        """
        Handle the end of an iteration.
        """
        total_time = format_time(time.time() - self.start_t)
        self.comment = f'100% [{self.total}/{self.total}] [{total_time}]'
        self.on_update(self.total, self.comment, self.postfix, False, 1)
        self.display = False
        if not self.in_jupyter:
            print('\n')

    def on_update(self, val, comment='', postfix='', interrupted=False, up=1):
        """
        Update the progress bar.

        Args:
            val: Current value of the progress.
            comment: Comment to display.
            postfix: Additional information to display.
            interrupted: Whether the process was interrupted.
            up: Number of lines to move up.
        """
        if not self.display:
            return
        if self.in_jupyter:
            # Update HTML progress bar in Jupyter environment
            self.progress = html_progress_bar(val, self.total, comment, postfix, interrupted)
            self.out.update(HTML(self.progress))
        else:
            if self.comment_tail:
                comment = comment + f' [{self.comment_tail}]'
            progress = text_progress_bar(val, self.total, comment, postfix, interrupted)
            print(move_up * up + progress, end='')

    def on_interrupt(self, msg='interrupted'):
        """
        Handle the case of interruption.

        Args:
            msg: Message to display for the interruption.
        """
        comment = self.comment + f' [{msg}]' if msg else self.comment
        self.on_update(self.last_v, comment, self.postfix, interrupted=True, up=1)
        if not self.in_jupyter:
            print('\n')

    def __iter__(self):
        """
        Iterate over the provided generator.

        Yields:
            Output from the generator.
        """
        if self.total != 0:
            self.update(0)
        try:
            for i, o in enumerate(self.gen):
                if self.total and i >= self.total:
                    break
                yield o
                self.update(i + 1)
            if self.total is None and self.last_v is not None:
                self.total = i + 1
                self.update(self.total)
        except Exception as e:
            self.on_interrupt()
            raise e

    def update_bar(self, val):
        """
        Update the progress bar details.

        Args:
            val: Current value of the progress.
        """
        if self.total == 0:
            return self.on_update(0, '100% [0/0]')
        if val == 0:
            self.comment = f'0% [0/{self.total}]'
            return self.on_update(0, self.comment)
        pct = '' if self.total is None else f'{100 * val / self.total:.2f}%'
        tot = '?' if self.total is None else str(self.total)
        elapsed_t = self.last_t - self.start_t
        remaining_t = '?' if self.pred_t is None else format_time(self.pred_t - elapsed_t)
        elapsed_t = format_time(elapsed_t)
        self.comment = f'{pct} [{val}/{tot}] [{elapsed_t}{self.lt}{remaining_t}]'
        self.on_update(val, self.comment, self.postfix)

    def set_postfix(self, **kwargs):
        """
        Set the postfix information for the progress bar.

        Args:
            **kwargs: Additional information to display.
        """
        if not self.display:
            return
        postfix = ''
        if 'i' in kwargs and 'n' in kwargs:
            from tqdm.std import Bar
            i, n = kwargs['i'], kwargs['n']
            kwargs.pop('i')
            kwargs.pop('n')
            ratio = i / n
            postfix += format(Bar(ratio, default_len=20))
            postfix += f'{100 * i/n:.2f}%'
            postfix += f' [{i}/{n}]'
        if kwargs:
            postfix += ' ['
            for i, (key, value) in enumerate(kwargs.items()):
                if isinstance(value, float):
                    postfix = postfix + f'{key}={format_number(value)}, '
                else:
                    postfix = postfix + f'{key}={value}, '
            postfix = postfix[:-2] + ']'
        self.postfix = postfix
        self.on_update(self.last_v, self.comment, self.postfix)
