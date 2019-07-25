import tkinter as tk
from tkinter import messagebox
from tkinter import font

from search import SimilaritySearch


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.tk.call('tk', 'scaling', 1.4)
        def_font = font.Font(family='newspaper', size=14)
        self.master.option_add("*Font", def_font)
        self.grid()
        self.create_entries()
        self.create_description()

        self.search = SimilaritySearch('./neural_hashes.csv', './dataset.csv')

        self.translation = {'similar': 'podobne', 'dissimilar': 'niepodobne'}
        self.radio_mapping = {1: 'similar', 0: 'dissimilar'}

    def create_description(self):
        self.description = tk.Label(text='Aplikacja do wyszukiwania filmów '
                                         'podobnych oraz niepodobnych\n'
                                         'na podstawie skrótów nadanych '
                                         'przez sieć neuronową\n realizującą '
                                         'funkcje mieszające')
        self.description.grid(row=3, columnspan=2, pady=15)

    def create_entries(self):
        """Creates elements in main window."""
        self.query_label = tk.Label(self, text='Zapytanie (film)')
        self.query_label.grid(row=1, column=0, padx=10, pady=10)
        self.query_entry = tk.Entry(self)
        self.query_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

        self.k_label = tk.Label(self, text='Liczba wyników')
        self.k_label.grid(row=2, column=0, padx=10, pady=10)
        self.k_entry = tk.Entry(self)
        self.k_entry.grid(row=2, column=1, columnspan=2, padx=10, pady=10)

        self.run_button = tk.Button(self, text='Szukaj')
        self.run_button['command'] = self.process_query
        self.run_button.grid()

        self.type_var = tk.IntVar()
        self.type_radio = dict()
        self.type_radio['similar'] = tk.Radiobutton(
            self,
            text='podobne',
            variable=self.type_var, value=1).grid(row=1, column=3,
                                                  padx=10, pady=10)

        self.type_radio['dissimilar'] = tk.Radiobutton(
            self,
            text='niepodobne',
            variable=self.type_var, value=0).grid(row=2, column=3,
                                                  padx=10, pady=10)


    def process_query(self):
        """Processes the data from query and returns result."""
        name = self.query_entry.get()
        try:
            k = int(self.k_entry.get())
        except Exception:
            messagebox.showinfo('Niepoprawna wartość k',
                                f'Wartość k powinna mieścić się w zakresie '
                                f'(1, 20)')
            return 0

        # validation
        if k <= 0 or k > 20:
            # show prompt
            messagebox.showinfo('Niepoprawna wartość k',
                                f'Wartość k powinna mieścić się w zakresie '
                                f'(1, 20)')
            return 0

        if not self.search.set_name(name):
            messagebox.showinfo('Brak filmu w bazie.',
                                f'Film {name} nie występuje w bazie.')
            return 0

        items = self.search.get_k_items(name, k)
        query_h = self.search.get_hash(name)
        # show results as window
        similarity_type = self.radio_mapping[int(self.type_var.get())]
        # for similarity_type in items:
        self.present_results(tk.Toplevel(self.master),
                             items[similarity_type], query_h,
                             similarity_type=similarity_type)

    def present_results(self, window, items, query_h, similarity_type=None):
        window.title(f'Filmy {self.translation[similarity_type]}')
        headers = ('Numer', 'Film', 'Indeks Jaccarda', 'Odległość Hamminga')
        for i, name in enumerate(headers):
            temp = tk.Label(window, text=name)
            temp.grid(row=0, column=i, padx=10, pady=10)

        offset = 1

        for i, movie in enumerate(items):
            values = [offset+i, movie, items[movie],cd 
                      self.search.num_diff(query_h,
                                           self.search.get_hash(movie))]
            for j, value in enumerate(values):
                temp = tk.Label(window, text=value)
                temp.grid(row=offset + i, column=j, padx=10, pady=10)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Wyszukiwarka filmów na podstawie podobieństwa')
    app = Application(master=root)
    app.mainloop()
