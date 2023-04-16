import math
import tkinter
import tkinter.ttk as ttk


class NumericSpinbox(tkinter.Frame):
    def __init__(self, parent, label, from_, to, default, command):
        tkinter.Frame.__init__(self, parent)

        spinbox_style = ttk.Style()
        spinbox_style.theme_use('default')
        spinbox_style.configure('Numeric.TSpinbox', arrowsize=47)

        self.__command = command
        self.__current_value = tkinter.IntVar(value=default)

        self.__label = tkinter.Label(self, text=label, anchor="w")
        self.__spinbox = ttk.Spinbox(self, style='Numeric.TSpinbox', from_=from_, to=to, state='readonly',
                                     width=math.floor(math.log10(to)+1), textvariable=self.__current_value,
                                     command=self.__value_changed)
        self.__spinbox.pack(padx=5, pady=0, side="left")

        self.__label.pack(side="left", fill="x")
        self.__spinbox.pack(side="right", fill="x", padx=4)

    def __value_changed(self):
        self.__command(self.__current_value.get())

    def enabled(self):
        self.__label.configure(state="normal")
        self.__spinbox.configure(state="readonly")

    def disabled(self):
        self.__label.configure(state="disabled")
        self.__spinbox.configure(state="disabled")
