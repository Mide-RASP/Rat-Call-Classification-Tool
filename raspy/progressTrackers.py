from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections


class ProgressTrackerBase(object):
    """ The base class for a generic progress tracking object. Handles tracking
        the number of subtasks completed for each of a set of nested tasks.
        Intended to be derived.
    """
    TaskDescriptor = collections.namedtuple(
        'TaskDescriptorBase',
        'subcount subtotal uniform'
    )

    def _updateSubTaskCount(self, subcount=None, increment=None):
        """ Updates the subtask counter for the given task. Used internally
            through `__call__` & `context` methods. Only one of the `subcount`
            and `increment` parameters should be used in a single call to this
            method.

            @param subcount: the value to which to set the current task's
                "subtasks completed" counter
            @param increment: the amount by which to increment the current
                task's "subtasks completed" counter
        """

        # short-circuit when not updating the subtask count
        if subcount is None and increment is None:
            return
        if len(self._task_stack) == 0:
            raise RuntimeError('no main task under which to update subtask count')

        task = self._task_stack[-1]
        if subcount is None:  # tries count first, then resorts to increment
            if increment < 0:
                raise RuntimeError('cannot decrement subtask count')
            subcount = task.subcount + increment

        if subcount > task.subtotal:
            raise ValueError('cannot update subtask count above preset total')

        if subcount != task.subcount:
            self._task_stack[-1] = task._replace(subcount=subcount)

    def _getProgressIndicativeTasks(self):
        """ Collects all tasks, in hierarchical order, that are indicative of
            overall task completeness.

            Whether or not a given task is taken depends on:
            - if its subtasks' execution time are defined relative to each
              other (i.e., with the `uniform` keyword argument)
            - if the number of subtasks is known in advance (i.e., with the
              `subtotal` keyword)
        """
        def generator():
            for task in self._task_stack:
                if not task.subtotal:
                    return
                if task.uniform:
                    yield task
                elif task.subtotal - task.subcount > 1:
                    return
        return tuple(generator())

    def _getPercentageEstimate(self, progressTasks=None):
        """ Calculates the "percentage completed" estimate for the given task
            completion state. Not used in this class; intended as a utility for
            derived classes.
        """
        import fractions

        if progressTasks is None:
            progressTasks = self._getProgressIndicativeTasks()

        ppwhole = fractions.Fraction(0)
        weight = fractions.Fraction(1)
        for task in progressTasks:
            subppwhole = fractions.Fraction(task.subcount, task.subtotal)
            ppwhole += subppwhole * weight
            weight /= task.subtotal

        return ppwhole

    # -------------------------------------------------------------------------

    def __init__(self):
        self._task_stack = []

    def __call__(self, count=None, increment=None):
        self._updateSubTaskCount(count, increment)

    def _enter(self, subtotal=None, uniform=False, **kwargs):
        self._task_stack.append(self.TaskDescriptor(0, subtotal, uniform, **kwargs))
        return self

    def _exit(self, type, value, traceback):
        self._task_stack.pop()

    @property
    def context(self):
        """ Creates a new task spanning the lifetime of the returned context
            manager. Subtasks are created by nesting context managers created
            by this method.

            The context manager uses the `self._enter` & `self._exit` methods
            to implement its `__enter__` & `__exit__` magic methods, which
            permits derived classes to override this behavior w/o rewriting
            this method.
        """
        class ContextManagerProxy(object):
            def __init__(self_proxy, *args, **kwargs):
                self_proxy.args = args
                self_proxy.kwargs = kwargs

            def __enter__(self_proxy):
                return self._enter(*self_proxy.args, **self_proxy.kwargs)

            def __exit__(self_proxy, *args, **kwargs):
                return self._exit(*args, **kwargs)

        return ContextManagerProxy


class ProgressPrinter(ProgressTrackerBase):
    """ The base class for a generic progress tracking object. Handles tracking
        the number of subtasks completed for each of a set of nested tasks.
        Intended to be derived.
    """
    TaskDescriptor = collections.namedtuple(
        'TaskDescriptor',
        ProgressTrackerBase.TaskDescriptor._fields + ('entryMessage', 'exitMessage')
    )

    def _print(self, message):
        """ Prints a generic message to stdout, inserting indentation based on
            nested task depth. Called internally by `__call__` & `context`.
        """
        if message is not None:
            indent = '  ' * (len(self._task_stack)-1)
            self._printFunc('{}{}'.format(indent, message))

    # -------------------------------------------------------------------------

    def __init__(self, printFunc=print):
        super(ProgressPrinter, self).__init__()
        self._printFunc = printFunc

    def __call__(self, message=None, count=None, increment=None):
        """ Modified from base implementation to print message on call.
        """
        self._print(message)
        return super(ProgressPrinter, self).__call__(count, increment)

    def _enter(self, entryMessage=None, exitMessage=None, subtotal=None,
               uniform=True):
        """ Modified from base implementation to print message on call.
        """
        self._print(entryMessage)
        return super(ProgressPrinter, self)._enter(
            subtotal, uniform,
            entryMessage=entryMessage,
            exitMessage=exitMessage
        )

    def _exit(self, xtype, value, traceback):
        """ Modified from base implementation to print message on call.
        """
        exitMessage = (self._task_stack[-1].exitMessage
                       if xtype is None else None)
        result = super(ProgressPrinter, self)._exit(xtype, value, traceback)
        self._print(exitMessage)
        return result
