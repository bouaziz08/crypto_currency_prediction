import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import scrolledtext as st
import time

class Interface:
    def __init__(self, master, intel,exit):
        self.master = master
        self.intel = intel
        self.mod = False
        self.exit = exit
        self.master.geometry("700x350")
        self.master.title("CMPA")
        self.master.iconbitmap("front_end/icon.ico")
        button_pressed = tk.StringVar()
        self.master.protocol("WM_DELETE_WINDOW", self.comfirm)

        frame = tk.Frame(self.master, border=1)
        frame.pack(pady=0, padx=0, fill="both", expand=True)


        # img = Image.open("front_end/xconvert.com (2).png")
        self.bg = ImageTk.PhotoImage(file = "front_end/background.png")
        label = tk.Label(master=frame,image=self.bg)
        label.place(x=0,y=0)

        # label = tk.Label(master=frame, text="User interface")
        # label.pack(pady=10, padx=10)

        label = tk.Label(master=frame, text="crypto_money")
        label.place(x=20, y=40)
        label.config(bg="#f6e58d")
        self.crypto_money = ttk.Combobox(master=frame, width=30, height=30)
        self.crypto_money.place(x=20, y=70)
        # crypto_money.pack(padx=10, pady=10)



        label = tk.Label(master=frame, text="money")
        label.place(x=420, y=40)
        label.config(bg="#f6e58d")
        self.currency = ttk.Combobox(master=frame, width=30, height=30)
        self.currency.place(x=420, y=70)
        # currency.pack(padx=10, pady=10)

        label = tk.Label(master=frame, text="Days")
        label.place(x=20, y=140)
        label.config(bg="#f6e58d")
        self.days = ttk.Combobox(master=frame, width=30, height=30)
        self.days.place(x=20, y=170)

        label = tk.Label(master=frame, text="Create new model")
        label.place(x=420, y=140)
        label.config(bg="#f6e58d")
        self.mod = ttk.Combobox(master=frame, width=30, height=30)
        self.mod.place(x=420, y=170)
        #days.pack(padx=10, pady=10)



        # resultat = self.clic()
        # label1 = tk.Label(master=frame)
        # label1.place(x=90, y=250)
        # label1.config(text="Today price : "+str(resultat))
        # label2 = tk.Label(master=frame)
        # label2.place(x=90, y=300)
        # label2.config(text="Price after prediction : "+str(resultat))

        #button.pack(padx=500, pady=600)


        self.crypto_money['values'] = ['BTC', 'ETH', 'DOGE']
        self.currency['values'] = ['USD', 'EUR', 'GBP']
        self.days['values'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.mod['values'] = [False,True]

        self.crypto_money.current(0)
        self.currency.current(0)
        self.days.current(0)
        self.mod.current(0)
        # progress = ttk.Style()
        # progress.theme_use('clam')
        # progress.configure("red.Horizontal.TProgressbar", background="#108cff")
        # progress = ttk.Progressbar(master , length=300, mode='determinate', style='red.Horizontal.TProgressbar')
        # progress.place(x=50, y=300)
        # button = tk.Button(master=frame, width=20, text="exit", command=self.comfirm)
        # button.place(x=50, y=450)



        button = tk.Button(master=frame, width=20, text="Predict", command=self.clic)
        button.place(x=450, y=225)
        # button.wait_variable(button_pressed)

    def clic(self):
        intel = [self.crypto_money.get(), self.currency.get(), self.days.get(), self.mod.get()]
        button_pressed = 1
        self.intel(intel)

    def comfirm(self):
        leave = messagebox.askyesno("confirmation", "Do you want to Exit ???", parent=self.master)
        if leave:
            self.master.destroy()
            self.exit()


# textarea = st.ScrolledText(frame, width=90, height=5)
# textarea.place(x=2, y=300)
#
# textarea.insert('insert', clic())

# master.protocol("WM_DELETE_master", comfirm)
# master.mainloop()
