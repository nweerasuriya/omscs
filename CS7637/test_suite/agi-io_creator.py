import json
import os.path
import uuid
import tkinter as tk
from copy import deepcopy

import ttkthemes
from tkinter import ttk, filedialog, messagebox, Canvas
from tkinter.ttk import Frame

from numpy import typing as npt
import numpy as np

from CreatorArcData import ArcData
from CreatorArcSet import ArcSet

UNFILLED = 'black'
BLUE = 'blue'
RED = 'red'
GREEN = 'green3'
YELLOW = 'yellow'
GREY = 'grey50'
MAGENTA = 'magenta'
ORANGE = 'orange2'
CYAN = 'cyan'
BROWN = 'saddle brown'

colors = (UNFILLED, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, CYAN, BROWN)

OUTLINE = 'DeepSkyBlue2'

INPUT = 'input'
OUTPUT = 'output'
DEFAULT_SIZE = 4


class Creator:
    def __init__(self, frame: ttkthemes.ThemedTk):
        self.last_directory = '.'

        ''' The menu bar for holding the themes '''
        menubar = tk.Menu(frame)
        frame.configure(menu=menubar)
        theme_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Theme', menu=theme_menu)
        self.default = tk.StringVar(value='Black')
        themes = frame.get_themes()
        sorted_themes = sorted(themes)

        for t in sorted_themes:
            theme_menu.add_radiobutton(label=str.capitalize(t), variable=self.default, value=str.capitalize(t),
                                       command=lambda: self.change_theme(str.lower(self.default.get())))

        def instructions():
            about_text = '''
            Themes menu will change the appearance of gui
            About/Usage will display this dialog
            
            Add IO Pair: adds a new set
            Remove IO Pair: removes the last set in list
            Use drop down to select IO Pair for editing
              changing IO Pair automatically saves changes
            
            Read File: open an Arc.json file
            Write File: save Arc.json file (auto generates names)
            
            Changing grid size (input/output) will reset grid
              min = 1x1  max = 30x30
              window will resize with grid changes
            
            Click color pallet to select color
              LMB click single cell to paint it
              CTRL-DRAG LMB will fill multiple cells
            
            Highlight Cells
              SHIFT-DRAG LMB to highlight multiple  cells
              ESC to clear selection
            
            Copy/Paste
              CTRL-C Copy selected cells
              CTRL-V Paste selected cells
              CTRL-P Special Paste selected cells
                  (dest cells not overwritten if copied cells are blank) 
                Highlight cells to copy then CTRL-C
                Highlight single cell (top left corner)
                  in input or output grids then CTRL-V (or CTRL-P)
                
            Rotating Selections (square sel only)
              ALT-7 Rotate CCW (i.e. 270 degrees)
              ALT-8 Rotate CW (i.e. 180 degrees)
              ALT-9 Rotate CW (i.e. 90 degrees)
            
            Boolean Operations: (must be equal size selections)
              Select operation from drop down menu
              Highlight 1st selection press 'Sel 1'
              Highlight 2nd selection press 'Sel 2'
              Select output color from Color Pallet
              Press 'Execute'  
                (resizes output, result displays in color selected)
              At any stage select 'Clear' to reset
              
            '''
            messagebox.showinfo("How To Use", about_text)

        about_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='About', menu=about_menu)
        about_menu.add_command(label="Usage", command=instructions)

        '''
        the dictionaries that hold the input/output Rectangle that are created to represent 
        what is seen on the screen
        '''
        self.input_cells: dict[str, int] = dict()  # the input rectangles keyed by row:col
        self.output_cells: dict[str, int] = dict()  # the output rectangles keyed by row:col

        self.cell_size: int = 0  # the width/height of an individual Rectangle (i.e. pixel)

        '''
        arc_data_sets is the backing store for the problem that is loaded in the application.
        this doesn't necessarily match with the data in the current_working_set
        '''
        self.arc_data_sets: dict[str, ArcSet] = dict()  # the backing store for the problem that is loaded

        '''
        The current_working_set is the data that is visible to the user in both input and output grids
        and doesn't necessarily match with the data in the arc_data_sets.
        
        When this gets changed to a new working set, the data should be written to the corresponding
        arc_data_set so that the backing store stays in sync with any changes made prior to the swap
        '''
        self.current_working_set: ArcSet = ArcSet(ArcData(np.zeros((DEFAULT_SIZE, DEFAULT_SIZE))),
                                                  ArcData(np.zeros((DEFAULT_SIZE, DEFAULT_SIZE))))

        self.current_set_name: str = ""  # the name in the combo box the user can see
        self.normalized_rotation = None  # used to store last known rotation

        self.ttk_style: ttk.Style = ttk.Style()

        frame.columnconfigure(index=0, weight=1)
        frame.rowconfigure(index=1, weight=1)

        self.ttk_style.configure('Main.TFrame', borderwidth=5, relief='ridge')
        main_frame = ttk.Frame(frame, style='Main.TFrame')
        main_frame.grid(row=0, column=0, padx=5, pady=5)

        # contains all the widgets for manipulating the arc-grids
        tool_bar = ttk.Frame(main_frame, style='Main.TFrame')
        tool_bar.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        # used to add training paris to the input-output set ---------------------------------------
        info_frame = ttk.Frame(tool_bar, style='Main.TFrame')
        info_frame.grid()

        btn_add_pair = ttk.Button(info_frame, text="Add IO Pair", command=self.add_pair)
        btn_add_pair.grid(padx=5, pady=5)
        btn_remove_pair = ttk.Button(info_frame, text="Remove IO Pair", command=self.remove_pair)
        btn_remove_pair.grid(padx=5, pady=5)
        ''' 
        create the initial data for the combo box
        create a read only combo_box and attach it to the info_frame panel
        set the combo in the third row of the info_panel
        set the currently selected index of the combo box
        set the function to trigger with combo box is changed
        '''
        self.options = ["Training 1", "Test"]
        self.arc_pair_combo = ttk.Combobox(info_frame, values=self.options, state='readonly')
        self.arc_pair_combo.grid(padx=5, pady=5)
        self.arc_pair_combo.current(0)
        self.arc_pair_combo.bind('<<ComboboxSelected>>', self.on_select)

        # -------------------------------------------------------------------------------------------
        #  TODO Here we need another tool panel for things like problem reset and/or Panel Reset?
        # -------------------------------------------------------------------------------------------

        # panel used to read in and/or write out IO sets
        read_write_frame = Frame(tool_bar, style='Main.TFrame')
        read_write_frame.grid(row=0, column=2, padx=5, pady=5)

        btn_read_file = ttk.Button(read_write_frame, text='Read File', command=self.read_io_file)
        btn_read_file.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        btn_write_file = ttk.Button(read_write_frame, text='Write File', command=self.write_io_file)
        btn_write_file.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.lbl_file_name = ttk.Label(read_write_frame, text='', width=25)
        self.lbl_file_name.grid(row=3, column=0, sticky='nsew', padx=5, pady=5)

        # the color palette that we can select different colors from
        self.color_pallet = tk.Canvas(tool_bar, width=452, height=47, background="grey25", highlightbackground="grey25")
        self.color_pallet.grid(columnspan=3, padx=5, pady=5)
        self.palette_rects = []
        self.make_color_palette()
        self.color_pallet.bind("<1>", self.pallet_clicked)

        # ics is the index of the currently selected color.
        self.color_index = 0
        self.select_pallet_color(self.color_index)

        # -------------------------------------------------------------------------------------------
        # panel to control input and output panels for the grid creation
        input_output_frame = Frame(tool_bar, width=150, height=150, style='Main.TFrame')
        input_output_frame.grid(row=0, column=3, sticky='nsew', padx=5, pady=5, rowspan=3)

        input_grid_info = Frame(input_output_frame, borderwidth=5)
        input_grid_info.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        lbl_input_grid = ttk.Label(input_grid_info, text='Input Grid Size')
        lbl_input_grid.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=3)

        self.entry_input_rows = ttk.Entry(input_grid_info, width=3)
        self.entry_input_rows.grid(row=1, column=0)
        self.entry_input_rows.insert(0, str(DEFAULT_SIZE))
        lbl_in_x = ttk.Label(input_grid_info, text='X')
        lbl_in_x.grid(row=1, column=1)
        self.entry_input_cols = ttk.Entry(input_grid_info, width=3)
        self.entry_input_cols.grid(row=1, column=2)
        self.entry_input_cols.insert(0, str(DEFAULT_SIZE))
        btn_resize_input = ttk.Button(input_grid_info, text='Resize/Reset', command=lambda: self.resize_grid(INPUT))
        btn_resize_input.grid(row=2, column=0, padx=5, pady=5, columnspan=3, sticky='nsew')

        output_grid_info = Frame(input_output_frame, width=150, height=150, borderwidth=5)
        output_grid_info.grid(row=0, column=1, padx=5, pady=5)
        lbl_output_grid = ttk.Label(output_grid_info, text='Output Grid Size')
        lbl_output_grid.grid(row=0, column=0, sticky="nsew", padx=5, pady=5, columnspan=3)

        self.entry_output_rows = ttk.Entry(output_grid_info, width=3)
        self.entry_output_rows.grid(row=1, column=0, padx=0, pady=0)
        self.entry_output_rows.insert(0, str(DEFAULT_SIZE))
        lbl_out_x = ttk.Label(output_grid_info, text='X')
        lbl_out_x.grid(row=1, column=1, padx=0, pady=0)
        self.entry_output_cols = ttk.Entry(output_grid_info, width=3)
        self.entry_output_cols.grid(row=1, column=2, padx=0, pady=0)
        self.entry_output_cols.insert(0, str(DEFAULT_SIZE))
        btn_resize_output = ttk.Button(output_grid_info, text='Resize/Reset', command=lambda: self.resize_grid(OUTPUT))
        btn_resize_output.grid(row=2, column=0, padx=5, pady=5, columnspan=3, sticky='nsew')

        btn_copy_in_to_out = ttk.Button(input_output_frame, text="Copy Input to Output",
                                        command=self.copy_input_to_output)
        btn_copy_in_to_out.grid(row=1, column=0, columnspan=2)

        self.error_label = ttk.Label(input_output_frame, text='')
        self.error_label.grid(row=2, column=0, columnspan=2)

        # -----------------------------------------------------------------------------------------------------------
        # boolean operations panel

        '''
            The boolean dictionary is used to store the highlighted selections for the operation
            The operation uses 2 unique selections of equal size from the input grid 
              and combines them based on the dropdown selection made, 
              resizing the output grid to the width and height of the of the combined selection
            The output color will be the currently selected output color bar selection.
            
            The dictionary itself will hold the selections made. 
                The keys should be simply 1 & 2
                The values are a List of coordinates and the color index number for that coordinate
                ex. [(('0','0'), int), (('1','1'), int)]
                
            As an aside maybe I just need to store the selection rectangles themselves?
        '''
        self.bool_op_dict: dict[int, np.ndarray] = dict()

        boolean_op_panel = Frame(tool_bar, width=150, height=150, style='Main.TFrame')
        boolean_op_panel.grid(row=0, column=4, sticky='nsew', padx=5, pady=5, rowspan=3)

        lbl_boolean = ttk.Label(boolean_op_panel, text="Boolean Operations")
        lbl_boolean.grid(row=0, column=0, padx=5, pady=5, columnspan=2, sticky='nsew')

        self.bool_options = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
        self.bool_op_combo = ttk.Combobox(boolean_op_panel, values=self.bool_options, state='readonly')
        self.bool_op_combo.grid(row=1, column=0, padx=5, pady=5, columnspan=2)
        self.bool_op_combo.current(0)

        self.btn_bool_sel_1 = ttk.Button(boolean_op_panel, text='Sel 1', command=lambda: self.bool_selection(1))
        self.btn_bool_sel_1.grid(row=2, column=0, padx=5, pady=5)

        self.btn_bool_sel_2 = ttk.Button(boolean_op_panel, text='Sel 2',
                                         command=lambda: self.bool_selection(2), state='disabled')
        self.btn_bool_sel_2.grid(row=2, column=1, padx=5, pady=5)

        self.btn_bool_execute = ttk.Button(boolean_op_panel, text='Execute',
                                           command=self.bool_op_execute, state='disabled')
        self.btn_bool_execute.grid(row=3, column=0, padx=5, pady=5, columnspan=2)

        self.btn_bool_clear = ttk.Button(boolean_op_panel, text='Clear',
                                         command=self.bool_op_clear, state='disabled')
        self.btn_bool_clear.grid(row=4, column=0, padx=5, pady=5, columnspan=2)

        # -- create the input and output grids ------------------------------------------------------------------

        grid_frame = Frame(main_frame, style='Main.TFrame')
        grid_frame.grid(row=1, column=0, padx=5, pady=5)

        self.input_grid_canvas = tk.Canvas(grid_frame, background='grey80', name="input_grid")
        self.input_grid_canvas.grid(row=0, column=0, padx=10, pady=10, sticky='n')
        self.input_grid_canvas.bind("<1>", self.input_grid_clicked)

        self.output_grid_canvas = Canvas(grid_frame, background="grey80", name='output_grid')
        self.output_grid_canvas.grid(row=0, column=2, padx=10, pady=10, sticky='n')
        self.output_grid_canvas.bind("<1>", self.output_grid_clicked)

        # end of gui creation stuff
        # -----------------------------------------------------------------------------------------------------

        # set up the initial grids
        self.resize_grid(INPUT)
        self.resize_grid(OUTPUT)
        self.create_initial_data()

        # ---Key Bindings over full application -------------------------------------------------------------
        frame.bind("<Escape>", lambda e: self.clear_highlights())

        # Copy Paste Delete Rotate Flip
        frame.bind("<Control-c>", self.copy_cells)
        frame.bind("<Control-v>", self.paste_cells)
        frame.bind("<Control-b>", self.paste_cells)
        frame.bind("<Delete>", self.delete_cells)

        frame.bind("<Alt-9>", self.rotate_selection)
        frame.bind("<Alt-8>", self.rotate_selection)
        frame.bind("<Alt-7>", self.rotate_selection)
        frame.bind("<Alt-v>", self.flip_selection)
        frame.bind("<Alt-h>", self.flip_selection)

        frame.bind("1", lambda e: self.select_pallet_color(1))
        frame.bind("2", lambda e: self.select_pallet_color(2))
        frame.bind("3", lambda e: self.select_pallet_color(3))
        frame.bind("4", lambda e: self.select_pallet_color(4))
        frame.bind("5", lambda e: self.select_pallet_color(5))
        frame.bind("6", lambda e: self.select_pallet_color(6))
        frame.bind("7", lambda e: self.select_pallet_color(7))
        frame.bind("8", lambda e: self.select_pallet_color(8))
        frame.bind("9", lambda e: self.select_pallet_color(9))
        frame.bind("0", lambda e: self.select_pallet_color(0))


        # -------------------------------------------------------------------------------------------
        self.copy_queue: list = list()

        # ---------------- Multiple box select/fill bindings ----------------------------------------

        self.select_box = None
        self.selected_objects = []
        self.orig_x = 0
        self.orig_y = 0

        self.input_grid_canvas.bind("<Control-1>", self.start_select)
        self.input_grid_canvas.bind("<Control-B1-Motion>", self.update_select)
        self.input_grid_canvas.bind("<Control-ButtonRelease-1>", self.stop_select)

        self.output_grid_canvas.bind("<Control-1>", self.start_select)
        self.output_grid_canvas.bind("<Control-B1-Motion>", self.update_select)
        self.output_grid_canvas.bind("<Control-ButtonRelease-1>", self.stop_select)

        self.input_grid_canvas.bind("<B1-Motion>", lambda e: self.clear_highlights("input"))
        self.output_grid_canvas.bind("<B1-Motion>", lambda e: self.clear_highlights("output"))

        # ---------------- Multiple box select/highlight for copy/paste/delete code --------------------------------
        self.hl_orig_x = 0
        self.hl_orig_y = 0
        self.hl_box = None
        self.hl_objects: dict[Canvas, list[int]] = dict()

        self.input_grid_canvas.bind("<Shift-1>", self.highlight_start_select)
        self.input_grid_canvas.bind("<Shift-B1-Motion>", self.highlight_update_select)
        self.input_grid_canvas.bind("<Shift-ButtonRelease-1>", self.highlight_stop_select)

        self.output_grid_canvas.bind("<Shift-1>", self.highlight_start_select)
        self.output_grid_canvas.bind("<Shift-B1-Motion>", self.highlight_update_select)
        self.output_grid_canvas.bind("<Shift-ButtonRelease-1>", self.highlight_stop_select)

    # --------------- End of Constructor calls ---------------------------------------------------------------

    def rotate_selection(self, event):
        pressed = event.keysym
        try:
            canvas, selected = self.hl_objects.popitem()
        except KeyError:
            messagebox.showerror("Rotate Error", "There is no selection highlighted")
            return

        normalized = self.normalize_selection(canvas, selected)

        # add 1 to each since indexes start at 0 and np array needs a size
        r, c = int(normalized[-1][0][0]) + 1, int(normalized[-1][0][1]) + 1
        c_list = [colors.index(x[1]) for x in normalized]
        mat: npt.NDArray = np.asarray(c_list).reshape((r, c))

        if pressed == '9':  # rotate 90 degrees  CCW 3x
            r_mat = np.rot90(mat, k=-1)
        elif pressed == '8':  # rotate 180 degrees  CCW 2x
            r_mat = np.rot90(mat, k=-2)
        elif pressed == '7':  # rotate 270 degrees  more rightly CCW 90 degrees
            r_mat = np.rot90(mat)
        else:
            raise Exception(f"Unknown rotation type pressed: {pressed}")

        if canvas.winfo_name().__contains__(INPUT):
            working_set = self.current_working_set.get_input_data()
            grid_rows = int(self.entry_input_rows.get())
            grid_cols = int(self.entry_input_cols.get())
        else:
            working_set = self.current_working_set.get_output_data()
            grid_rows = int(self.entry_output_rows.get())
            grid_cols = int(self.entry_output_cols.get())

        can_mat = np.asarray(canvas.find_all()).reshape((grid_rows, grid_cols))

        start_idx = list(zip(*np.where(can_mat == selected[0])))[0]
        r_selected_idx = list()
        for row in range(r_mat.shape[0]):
            for col in range(r_mat.shape[1]):
                r_selected_idx.append(tuple(np.add((row, col), start_idx)))

        r_selected = list()
        for x in r_selected_idx:
            try:
                r_selected.append(can_mat[x])
            except IndexError:
                continue

        # set existing selection to unfilled
        for rect in selected:
            canvas.itemconfig(rect, fill=UNFILLED, outline='')  # remove visual
            tag = canvas.gettags(rect)[0].split(':')
            working_set.data()[int(tag[0]), int(tag[1])] = colors.index(UNFILLED)  # remove from data set

        for rect, c_idx in zip(r_selected, r_mat.flatten()):
            canvas.itemconfig(rect, fill=colors[c_idx], outline='')
            tag = canvas.gettags(rect)[0].split(':')
            working_set.data()[int(tag[0]), int(tag[1])] = c_idx

        self.highlight_selected(canvas, r_selected)


    def flip_selection(self, event):
        pressed = event.keysym
        try:
            canvas, selected = self.hl_objects.popitem()
        except KeyError:
            messagebox.showerror("Flip Error", "There is no selection highlighted")
            return

        normalized = self.normalize_selection(canvas, selected)
        r, c = int(normalized[-1][0][0]), int(normalized[-1][0][1])
        r += 1
        c += 1
        c_list = [colors.index(x[1]) for x in normalized]
        arr = np.asarray(c_list).reshape((r, c))
        if pressed == 'h':
            flip = np.fliplr(arr)
        elif pressed == 'v':
            flip = np.flipud(arr)
        else:
            raise Exception(f"Unknown flip type pressed: {pressed}")

        grid_type = canvas.winfo_name()
        working_set = self.current_working_set.get_input_data() if grid_type == 'input_grid' \
            else (self.current_working_set.get_output_data())
        flat = flip.flatten()
        for rect, idx in zip(selected, flat):
            tag = canvas.gettags(rect)[0].split(':')
            canvas.itemconfig(rect, fill=colors[idx], outline='')
            working_set.data()[int(tag[0]), int(tag[1])] = idx

        self.highlight_selected(canvas, selected)

    def copy_cells(self, event):
        """
        Executes a copy of the data contained in the highlight dictionary.
        This removes the data from the dictionary, clears the outline in
        the selected rectangles and stores just the color and cell location
        in list (i.e. self.copy_queue).
        """
        '''
        We don't need to not copy the rectangles but the data they contain 
        i.e. the color they hold and their cell location.
        
        We also need to shift the position of the pseudo array to essentially 0:0
        so a paste of the data can be performed at any location adding the 
        starting row:col to each key to get the corresponding output rectangle
        '''
        canvas, selected = self.hl_objects.popitem()
        self.copy_queue.clear()  # clear the previously copied cells
        normalized = self.normalize_selection(canvas, selected)
        self.copy_queue.extend(normalized)

    @staticmethod
    def normalize_selection(canvas: Canvas, selected: list) -> list[tuple[tuple[str, str], str]]:
        """
        Normalize the list of rectangle coordinates to start at
        a 0:0 grid location that they can be used for varying
        tasks such as pasting, rotating, flipping etc.
        """
        local = list()
        first_tag = canvas.gettags(selected[0])[0]
        r, c = first_tag.split(':')
        sr = int(r)
        sc = int(c)
        for rect in selected:
            canvas.itemconfig(rect, outline='')
            fill = canvas.itemcget(rect, 'fill')
            tag = canvas.gettags(rect)[0]
            rect_r, rect_c = tag.split(":")
            row = int(rect_r) - sr
            col = int(rect_c) - sc
            local.append(((str(row), str(col)), fill))
        return local

    @staticmethod
    def selection_to_array(selection: list[tuple[tuple[str, str], str]]) -> np.ndarray:
        """
        Returns a numpy array from the current highlight selection
        """
        r, c = int(selection[-1][0][0]), int(selection[-1][0][1])
        r += 1
        c += 1
        c_list = [colors.index(x[1]) for x in selection]
        mat = np.asarray(c_list).reshape([r, c])
        return mat

    def paste_cells(self, event):
        """
        Fills the cells starting at a highlighted cell with the data contained
        in the self.copy_queue list. If more that just a single cell is highlighted,
        then the lowest ordered cell is the starting cell for the paste operation.

        If the copy_queue list is empty or the there are no current highlighted
        cells (i.e. the self.hl_objects dictionary is empty) this method will
        just return. Any data currently stored in the copy_queue will not be deleted
        (in case the user didn't select a destination cell yet) and even after
        a successful insertion the copy_queue will remain intact.
        """

        if len(self.copy_queue) <= 0:
            self.clear_highlights()
            return
        if len(self.hl_objects) <= 0:
            return
        local = deepcopy(self.copy_queue)  # we don't clear because we should be able to paste as many as we want
        canvas, selected = self.hl_objects.popitem()
        self.reset_highlights(canvas, selected)

        if canvas.winfo_name().__contains__(INPUT):
            grid_set = self.input_cells
            working_set = self.current_working_set.get_input_data()
        else:
            grid_set = self.output_cells
            working_set = self.current_working_set.get_output_data()

        start_cell = canvas.gettags(selected[0])[0]
        canvas.itemconfig(start_cell, outline='')
        sr, sc = start_cell.split(':')
        start_r = int(sr)  # start row
        start_c = int(sc)  # start column

        for cell in local:
            c = cell[0]
            cr = int(c[0])  # current row
            cc = int(c[1])  # current column
            dest_row = cr + start_r
            dest_col = cc + start_c
            # so here if the copied cell is 0 && keysym == 'p' skip filling the pixel
            if event.keysym == 'b' and cell[1] == UNFILLED:
                continue
            key = f'{dest_row}:{dest_col}'
            if not grid_set.keys().__contains__(key): continue  # skip if the new data will be written off the grid
            value = cell[1]
            rect = grid_set[key]
            canvas.itemconfig(rect, fill=value)
            working_set.data()[dest_row, dest_col] = colors.index(value)

    @staticmethod
    def reset_highlights(canvas: Canvas, highlights: list):
        """
        Clear the highlights from the list of supplied highlights
        """
        for r in highlights:
            canvas.itemconfig(r, outline='')

    def clear_highlights(self, grid=None):
        """
        Clears the highlight dictionary and removes
        the highlight outline for any cells that
        are currently stored. If any of the selected
        cells contain a color this will not remove those
        existing colors.
        """

        if grid == 'input':
            self.clear_hl_box(self.input_grid_canvas)
            return
        elif grid == 'output':
            self.clear_hl_box(self.output_grid_canvas)
            return

        while len(self.hl_objects) > 0:
            canvas, selected = self.hl_objects.popitem()
            for r in selected:
                canvas.itemconfig(r, outline='')

    def clear_hl_box(self, canvas: Canvas):
        canvas.delete(self.hl_box)
        self.hl_box = None
        self.hl_orig_x = 0
        self.hl_orig_y = 0

    def delete_cells(self, event):
        """
        Clears the highlight dictionary and removes
        the highlight outline for any cells that
        are currently stored AND sets the rectangle
        color to UNFILLED (i.e. black).
        """
        canvas, selected = self.hl_objects.popitem()
        if canvas.winfo_name().__contains__(INPUT):
            grid_data = self.current_working_set.get_input_data()
        else:
            grid_data = self.current_working_set.get_output_data()

        for rect in selected:
            canvas.itemconfig(rect, fill=UNFILLED, outline='')  # remove visual
            tag = canvas.gettags(rect)[0].split(':')
            grid_data.data()[int(tag[0]), int(tag[1])] = colors.index(UNFILLED)  # remove from data set

    def highlight_start_select(self, event):
        """
        Called when the user Shift+Clicks in a cell.
        Used to highlight 1 or more rectangles.
        """
        canvas = self.input_grid_canvas if event.widget.winfo_name() == 'input_grid' else self.output_grid_canvas
        self.hl_orig_x = canvas.canvasx(event.x)
        self.hl_orig_y = canvas.canvasy(event.y)

        self.hl_box = canvas.create_rectangle(self.hl_orig_x, self.hl_orig_y, self.hl_orig_x, self.hl_orig_y,
                                              outline=OUTLINE, width=2)

    def highlight_update_select(self, event):
        """
        Called when the user Shift+Clicks in a cell and drags the mouse.
        This will create a highlight rectangle to visually indicate how many
         grid cells are being selected.
        """
        if self.hl_box is None:
            return

        canvas = self.input_grid_canvas if event.widget.winfo_name() == 'input_grid' else self.output_grid_canvas
        hl_end_x = canvas.canvasx(event.x)
        hl_end_y = canvas.canvasy(event.y)
        if hl_end_x < self.hl_orig_x and hl_end_y < self.hl_orig_y:
            canvas.coords(self.hl_box, hl_end_x, hl_end_y, self.hl_orig_x, self.hl_orig_y)
        elif hl_end_x < self.hl_orig_x:
            canvas.coords(self.hl_box, hl_end_x, self.hl_orig_y, self.hl_orig_x, hl_end_y)
        elif hl_end_y < self.hl_orig_y:
            canvas.coords(self.hl_box, self.hl_orig_x, hl_end_y, hl_end_x, self.hl_orig_y)
        else:
            canvas.coords(self.hl_box, self.hl_orig_x, self.hl_orig_y, hl_end_x, hl_end_y)

    def highlight_stop_select(self, event):
        """
        Called when the user started a selection with Shift+Click and drags the mouse and then releases the mouse.
        This will find all the grid cells located under the highlight rectangle that was created, deletes the
        selection rectangle, and calls the self.highlight_selected method with the canvas and list of selected cells.
        """
        ctrl = (event.state & 0x4) != 0
        alt = (event.state & 0x8) != 0 or (event.state & 0x80) != 0
        shift = (event.state & 0x1) != 0

        canvas = self.input_grid_canvas if event.widget.winfo_name() == 'input_grid' else self.output_grid_canvas

        x1, y1, x2, y2 = canvas.coords(self.hl_box)

        canvas.delete(self.hl_box)
        selected: set[int] = set()

        selected.update(canvas.find_overlapping(x1, y1, x2, y2))
        # create a highlight for each box in the selected list
        self.highlight_selected(canvas, list(selected))

    def highlight_selected(self, canvas: Canvas, selected: list[int]):
        """
        Creates an outline around each selected rectangle in the provided canvas
        and stores this is a single use highlight dictionary using the canvas
        as the key and a sorted list of selected rectangles as the value
        (values should be row/col order sort).

        This does not perform any other actions other than store the highlighted
        cells that are used by other methods such as self.delete_cells(event),
        self.copy_cells(event), and self.paste_cells(self).
        """
        self.clear_highlights()
        cells = self.input_cells if canvas.winfo_name().__contains__(INPUT) else self.output_cells

        selected.sort()
        for s in selected:
            key = canvas.gettags(s)[0]
            rect = cells[key]
            canvas.itemconfig(rect, outline=OUTLINE, width=3)
        self.hl_objects[canvas] = list(selected)

    def start_select(self, event):
        """
        Called when the user CTRL+Clicks in a cell.
        Used to fill 1 or more rectangles with the same color
         at one time.
        """
        canvas = self.input_grid_canvas if event.widget.winfo_name() == 'input_grid' else self.output_grid_canvas
        self.orig_x = canvas.canvasx(event.x)
        self.orig_y = canvas.canvasy(event.y)

        self.select_box = canvas.create_rectangle(self.orig_x, self.orig_y, self.orig_x, self.orig_y,
                                                  outline='OliveDrab1', width=2)

    def update_select(self, event):
        """
        Called when the user started a selection with CTRL+Click and drags the mouse.
        This will create a selection rectangle to visually indicate how many
         grid cells are being selected for a color fill.
        """
        canvas = self.input_grid_canvas if event.widget.winfo_name() == 'input_grid' else self.output_grid_canvas

        end_x = canvas.canvasx(event.x)
        end_y = canvas.canvasy(event.y)
        if end_x < self.orig_x and end_y < self.orig_y:
            canvas.coords(self.select_box, end_x, end_y, self.orig_x, self.orig_y)
        elif end_x < self.orig_x:
            canvas.coords(self.select_box, end_x, self.orig_y, self.orig_x, end_y)
        elif end_y < self.orig_y:
            canvas.coords(self.select_box, self.orig_x, end_y, end_x, self.orig_y)
        else:
            canvas.coords(self.select_box, self.orig_x, self.orig_y, end_x, end_y)

    def stop_select(self, event):
        """
        Called when the user started a selection with CTRL+Click and drags the mouse and then releases the mouse.
        This will find all the grid cells located under the selection rectangle that was created, deletes the
        selection rectangle, and calls the self.fill_selected method with the canvas and list of selected cells
        """
        canvas = self.input_grid_canvas if event.widget.winfo_name() == 'input_grid' else self.output_grid_canvas

        x1, y1, x2, y2 = canvas.coords(self.select_box)
        canvas.delete(self.select_box)
        selected: set[int] = set()

        selected.update(canvas.find_overlapping(x1, y1, x2, y2))
        self.fill_selected(list(selected), canvas)

    def fill_selected(self, selected: list[int], canvas: Canvas):
        """
        Fills each of the selected cells with the appropriate color
        and updates the appropriate working set.
        """
        if canvas.winfo_name().__contains__(INPUT):
            working_set = self.current_working_set.get_input_data()
            cells = self.input_cells
        else:
            working_set = self.current_working_set.get_output_data()
            cells = self.output_cells

        for s in selected:
            key = canvas.gettags(s)[0]
            rect = cells[key]
            canvas.itemconfig(rect, fill=colors[self.color_index])
            coords = key.split(':')
            row = int(coords[0])
            col = int(coords[1])
            working_set.data()[row, col] = self.color_index

    # -----------------------------------------------------------------------------------------------------

    def on_select(self, event):
        """
        The function that gets triggered when selecting the combo box of Arc IO Pairs
        This should load the current IO pair into the grids and save off the existing data
        """
        # value = self.arc_pair_combo.current()          # index position in combo box
        new_set_name = self.arc_pair_combo.get()  # the name in the combo box
        # store current data in backing set
        self.arc_data_sets[self.current_set_name] = deepcopy(self.current_working_set)

        # make a deep copy of our backing set data for the new working set
        new_working_set: ArcSet = deepcopy(self.arc_data_sets[new_set_name])
        self.reset_entries_and_data(new_set_name, new_working_set)

    def reset_entries_and_data(self, new_name: str, new_data: ArcSet):
        """
        Reset the input and output Entry boxes when a new working set has been initiated,
        updates both the input and output grids and cells
        and sets the current working set
        """

        # set the row/col in the gui text boxes for input
        in_rows, in_cols = new_data.get_input_data().shape()
        self.entry_input_rows.delete(0, tk.END)
        self.entry_input_cols.delete(0, tk.END)
        self.entry_input_rows.insert(0, str(in_rows))
        self.entry_input_cols.insert(0, str(in_cols))
        # set the row/col in the gui text boxes for output
        out_rows, out_cols = new_data.get_output_data().shape()
        self.entry_output_rows.delete(0, tk.END)
        self.entry_output_cols.delete(0, tk.END)
        self.entry_output_rows.insert(0, str(out_rows))
        self.entry_output_cols.insert(0, str(out_cols))

        self.resize_grid(INPUT)
        self.resize_grid(OUTPUT)

        self.fill_grid(new_data.get_input_data().data(), INPUT)
        self.fill_grid(new_data.get_output_data().data(), OUTPUT)

        self.current_set_name = new_name
        self.current_working_set = new_data

    def add_pair(self):
        """
        Adds the tag to the combo box for a new training pair
        and adds an empty set to the backing data
        """
        # lists are zero index, and we want the second from the last, TEST pair is last,
        #  so we can just use the length of the list for the next number to add
        #  but will insert it in the second to the last position
        idx = len(self.options)
        io_pair_name = self.add_to_combo_box(idx)

        default_input = self.create_arc_data(6, 6)
        default_output = self.create_arc_data(6, 6)
        self.arc_data_sets[io_pair_name] = ArcSet(default_input, default_output)

    def add_to_combo_box(self, idx):
        """
        Adds a Training IO Pair name to the combo box.
        This does not add any data to the backing store
        """
        pair_name = f'Training {idx}'
        self.options.insert(len(self.options) - 1, pair_name)
        self.arc_pair_combo['value'] = self.options
        return pair_name

    def remove_pair(self):
        """
        Removes the last training pair from the combo box if there are more than just 1.
        Will not remove the Test or the first Training data (must always have at least these 2).
        Also removes the backing data set.
        """
        if len(self.options) == 2:
            return

        value = self.arc_pair_combo.current()  # index position in combo box
        if len(self.options) - 2 == value:
            first_set_name = 'Training 1'
            self.current_working_set = deepcopy(self.arc_data_sets[first_set_name])
            self.current_set_name = first_set_name
            self.arc_pair_combo.current(0)
            self.arc_pair_combo.event_generate("<<ComboboxSelected>>")

        # list is zero index, so we want the second to the last entry (last entry is Test)
        removed = self.options.pop(len(self.options) - 2)  # remove last training and not test
        self.arc_pair_combo['value'] = self.options
        self.arc_data_sets.pop(removed)

    def add_to_backing_store(self, key: str, data_set: ArcSet):
        """
        Adds a copy of the ArcSet to the backing store dictionary with the key name
        """
        self.arc_data_sets[key] = deepcopy(data_set)

    def make_color_palette(self):
        """
        Create the on-screen pallet of colors to chose from
        """
        pad = 5
        width = 40
        height = 40
        for i in range(len(colors)):
            x, y = pad * (i + 1) + i * width, pad
            rect = self.color_pallet.create_rectangle(x, y, x + width, y + height,
                                                      outline="black", fill=colors[i], tags=str(i))
            self.palette_rects.append(rect)

    def pallet_clicked(self, event):
        """Get the closest rectangle clicked within the color pallet."""
        near_tags = self.color_pallet.find_closest(event.x, event.y)
        for tag in near_tags:
            self.select_pallet_color(int(self.color_pallet.gettags(tag)[0]))

    def select_pallet_color(self, idx: int):
        """Select the color indexed at i in the colors list and highlights it in the pallet"""
        # remove border from previous selection
        self.color_pallet.itemconfig(self.palette_rects[self.color_index], outline='black', width=1)

        # set the new index & put a border around the selection
        self.color_index = idx
        self.color_pallet.itemconfig(self.palette_rects[self.color_index], outline='white', width=2)

    def read_io_file(self):
        file_path = filedialog.askopenfilename(parent=root, initialdir=self.last_directory,
                                               filetypes=[('JSON,', '*.json')])
        if not file_path:
            return

        self.last_directory = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)

        self.lbl_file_name['text'] = file_name

        # clear out all the old stuff
        self.arc_data_sets.clear()
        self.options.clear()
        self.input_cells.clear()
        self.output_cells.clear()
        self.clear_highlights()
        self.current_working_set = None
        self.input_grid_canvas.delete('all')
        self.output_grid_canvas.delete('all')
        self.arc_pair_combo['values'] = list()

        with open(os.path.join(file_path)) as p:
            flat_data: dict[str, dict] = json.load(p)
            trn_data: list[ArcSet] = list()
            for dt in flat_data['train']:
                d_input = ArcData(np.array(dt['input']))
                d_output = ArcData(np.array(dt['output']))
                trn_set: ArcSet = ArcSet(arc_input=d_input, arc_output=d_output)
                trn_data.append(trn_set)

            tst_data: list[ArcSet] = list()
            for tst in flat_data['test']:
                t_input = ArcData(np.array(tst['input']))
                t_output = ArcData(np.array(tst['output']))
                tst_set: ArcSet = ArcSet(arc_input=t_input, arc_output=t_output)
                tst_data.append(tst_set)

            for idx, training_set in enumerate(trn_data, 1):
                key = f"Training {idx}"
                self.options.append(key)
                self.add_to_backing_store(key, training_set)

            key = 'Test'
            self.options.append(key)
            self.add_to_backing_store(key, tst_data[0])

            self.arc_pair_combo['value'] = self.options

            first_set_name = 'Training 1'
            self.current_working_set = deepcopy(self.arc_data_sets[first_set_name])
            self.current_set_name = first_set_name
            self.arc_pair_combo.current(0)
            self.arc_pair_combo.event_generate("<<ComboboxSelected>>")

    def write_io_file(self):
        # save current to
        current_set_name = self.current_set_name
        # save off what is seen on screen
        self.arc_data_sets[current_set_name] = deepcopy(self.current_working_set)

        # flatten current data
        train_data = list()
        test_data = list()
        for key in self.arc_data_sets:
            if key.__contains__("Training"):
                train_data.append(self.arc_data_sets[key].to_dict())
            else:
                test_data.append(self.arc_data_sets[key].to_dict())

        flat_data = dict()
        flat_data['train'] = train_data
        flat_data['test'] = test_data
        data_str = json.dumps(flat_data)

        uid = uuid.uuid3(uuid.NAMESPACE_OID, data_str).__str__().split('-')[0]

        file_path = filedialog.asksaveasfilename(parent=root, defaultextension='.json', initialfile=str(uid),
                                                 initialdir=self.last_directory, filetypes=[('JSON,', '*.json')])
        if not file_path:
            return

        file_name = os.path.basename(file_path)
        self.last_directory = os.path.dirname(file_path)
        self.lbl_file_name['text'] = file_name
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(data_str)
            except Exception as e:
                print(f'Error writing file: {str(e)}')

    def input_grid_clicked(self, event):
        """
        Triggered when someone clicks inside the input grid.
        This method just checks to see what the closest rectangle
        that was clicked near and passes that info on to the
        fill_input_cell(key) method
        """
        near_tags = self.input_grid_canvas.find_closest(event.x, event.y)
        for tag in near_tags:
            try:
                key = self.input_grid_canvas.gettags(tag)[0]
            except KeyError:
                return

            # check if we are close enough to fill it in
            rect = self.input_cells[key]
            x1, y1, x2, y2 = self.input_grid_canvas.coords(rect)

            if x1 < event.x < x2 and y1 < event.y < y2:
                self.fill_cell(key, INPUT)

    def output_grid_clicked(self, event):
        """
        Triggered when someone clicks inside the output grid.
        This method just checks to see what the closest rectangle
        that was clicked near and passes that info on to the
        fill_output_cell(key) method
        """
        near_tags = self.output_grid_canvas.find_closest(event.x, event.y)
        for tag in near_tags:
            key = self.output_grid_canvas.gettags(tag)[0]

            # check if we are close enough to fill it in
            rect = self.output_cells[key]
            x1, y1, x2, y2 = self.output_grid_canvas.coords(rect)

            if x1 < event.x < x2 and y1 < event.y < y2:
                self.fill_cell(key, OUTPUT)

    def fill_cell(self, key: str, grid_type: str):
        """
        Changes the rectangle color
        specified by the supplied key.
        """
        if grid_type.__eq__(INPUT):
            cells = self.input_cells
            grid = self.input_grid_canvas
            data = self.current_working_set.get_input_data().data()
        elif grid_type.__eq__(OUTPUT):
            cells = self.output_cells
            grid = self.output_grid_canvas
            data = self.current_working_set.get_output_data().data()
        else:
            print(f'Unknown grid type passed to fill cells: {grid_type}')
            return

        rect: int = cells[key]
        grid.itemconfig(rect, fill=colors[self.color_index])

        # NOTE this is using a creator version of the ArcData that
        #  we can write to since the real ArcData does a deepcopy
        #  of the internal data
        keys = key.split(':')
        row = int(keys[0])
        col = int(keys[1])
        data[row, col] = self.color_index

    def fill_grid(self, data: np.ndarray, grid_type: str):
        """
        Refills the grid (input/output) with the data from the array
        """
        if grid_type.__eq__(INPUT):
            cells = self.input_cells
            grid = self.input_grid_canvas
        elif grid_type.__eq__(OUTPUT):
            cells = self.output_cells
            grid = self.output_grid_canvas
        else:
            print(f'Unknown grid type passed to fill grid method: {grid_type}')
            return

        for key in cells.keys():
            keys = key.split(':')
            row = int(keys[0])
            col = int(keys[1])
            rect: int = cells[key]
            grid.itemconfig(rect, fill=colors[data[row, col]], outline='')

    def resize_grid(self, grid_type: str):

        """
        Called when someone smashes the Resize button in the toolbar for the Output Grid Size
        or Called when the IOPair combo box is changed to a new ArcSet

        This does some checks to make sure there are numbers entered into the entry boxes and
        that the numbers are in the range of 1 to 30.

        **CAUTION: This also currently erases any data previously stored in the grid without warning.
        """
        if grid_type.__eq__(INPUT):
            entry_rows = self.entry_input_rows
            entry_cols = self.entry_input_cols
            cells: dict[str, int] = self.input_cells
            current_data = self.current_working_set.get_input_data()
        elif grid_type.__eq__(OUTPUT):
            entry_rows = self.entry_output_rows
            entry_cols = self.entry_output_cols
            cells: dict[str, int] = self.output_cells
            current_data = self.current_working_set.get_output_data()
        else:
            print(f'Unknown grid type passed to resize grid: {grid_type}')
            return

        try:
            rows = int(entry_rows.get())
        except ValueError:
            # not a number
            self.error_label['text'] = f'row {grid_type} is not a number (1-30)'
            return

        try:
            columns = int(entry_cols.get())
        except ValueError:
            # not a number
            self.error_label['text'] = f'column {grid_type} is not a number (1-30)'
            return

        if rows < 1 or rows > 30:
            self.error_label['text'] = f'{grid_type} rows must be between 1 and 30'
            return
        if columns < 1 or columns > 30:
            self.error_label['text'] = f'{grid_type} columns must be between 1 and 30'
            return

        self.error_label['text'] = ''
        cells.clear()
        current_data.reset_data(np.zeros((rows, columns), dtype=int))
        self.make_grid(rows, columns, grid_type)

    def make_grid(self, rows: int, columns: int, grid_type: str):
        """
        Makes the physical output grid rectangles and tags them accordingly
        so when they are clicked we can properly retrieve them from the
        backing store.
        """
        if grid_type.__eq__(INPUT):
            grid: Canvas = self.input_grid_canvas
            cells: dict[str, int] = self.input_cells
        elif grid_type.__eq__(OUTPUT):
            grid = self.output_grid_canvas
            cells = self.output_cells
        else:
            print(f'Unknown Type passed to resize grid: {grid_type}')
            return

        # determine the size of our blocks, this depends on the max of rows and columns
        max_cells = max(rows, columns)
        if max_cells <= 15:
            cell_size = 40
        elif max_cells <= 20:
            cell_size = 30
        else:
            cell_size = 20

        self.cell_size = cell_size
        pad = 4
        grid.delete('all')
        cells.clear()
        for row in range(rows):
            for col in range(columns):
                x_pad, y_pad = pad * (col + 1), pad * (row + 1)
                x, y, = x_pad + col * cell_size, y_pad + row * cell_size
                key = f'{row}:{col}'
                rect = grid.create_rectangle(x, y, x + cell_size, y + cell_size, fill=UNFILLED, tags=key, outline='')
                cells[key] = rect

        row_height = cell_size * rows + (rows * 4)
        col_width = cell_size * columns + (columns * 4)
        grid.configure(width=col_width, height=row_height)

    def create_initial_data(self):
        """
        Creates the initial data for 1 training ArcSet and 1 test ArcSet
        and stores it in the arc_data_sets dictionary as well as sets
        the current_working_set to the first training set.
        """
        # Training 1
        rows = int(self.entry_input_rows.get())
        cols = int(self.entry_input_cols.get())
        train1_in = self.create_arc_data(rows, cols)
        rows = int(self.entry_output_rows.get())
        cols = int(self.entry_output_cols.get())
        train1_out = self.create_arc_data(rows, cols)
        train1 = ArcSet(train1_in, train1_out)
        self.arc_data_sets["Training 1"] = train1
        self.current_working_set = deepcopy(train1)
        self.current_set_name = "Training 1"

        # Test
        rows = int(self.entry_input_rows.get())
        cols = int(self.entry_input_cols.get())
        test_in = self.create_arc_data(rows, cols)
        rows = int(self.entry_output_rows.get())
        cols = int(self.entry_output_cols.get())
        test_out = self.create_arc_data(rows, cols)
        test = ArcSet(test_in, test_out)
        self.arc_data_sets['Test'] = test

    @staticmethod
    def create_arc_data(rows: int, cols: int) -> ArcData:
        """
        Returns a np.ndarry in size rows, columns
        filled with all zeros (i.e. UNFILLED)
        """
        return ArcData(np.zeros((rows, cols), dtype=int))

    def copy_input_to_output(self):
        """
        Copies the input grid to the output grid.
        This will reset everything in the output grid
        and fill it with the same data as what is in the input
        """
        input_data = self.current_working_set.get_input_data().data()
        self.current_working_set.get_output_data().reset_data(deepcopy(input_data))

        self.output_grid_canvas.delete('all')
        self.output_cells.clear()

        # resize the output grid to match the input grid
        self.entry_output_rows.delete(0, tk.END)
        self.entry_output_rows.insert(0, self.entry_input_rows.get())
        self.entry_output_cols.delete(0, tk.END)
        self.entry_output_cols.insert(0, self.entry_input_cols.get())

        self.resize_grid(OUTPUT)
        self.update_grid(self.current_working_set.get_input_data().data(), OUTPUT)
        self.current_working_set.get_output_data().reset_data(
            deepcopy(self.current_working_set.get_input_data().data()))

    def update_grid(self, data: np.ndarray, grid_type: str):
        """
        Currently called from copy input. This does not resize
        the grid and will fail if the current input or output grid
        sizes are not the same size as the incoming np.ndarray.

        Call resize_grid with the new data's shape (rows, columns) which
        will resize and reset the data in the grid before calling this
        method.
        """
        if grid_type.__eq__(INPUT):
            for key in self.input_cells:
                keys = key.split(":")
                row = int(keys[0])
                col = int(keys[1])
                col_idx = data[row, col]
                rect: int = self.input_cells[key]
                self.output_grid_canvas.itemconfig(rect, fill=colors[col_idx])
        elif grid_type.__eq__(OUTPUT):
            for key in self.output_cells:
                keys = key.split(":")
                row = int(keys[0])
                col = int(keys[1])
                col_idx = data[row, col]
                rect: int = self.output_cells[key]
                self.output_grid_canvas.itemconfig(rect, fill=colors[col_idx])
        else:
            print(f'Update Grid unknown type. Expected {INPUT}/{OUTPUT} but got {grid_type} instead')

    def bool_selection(self, selection_number: int):
        """
        highlight cells then depending on the selection number store these in a bool_op dictionary
        """
        try:
            canvas, selected = self.hl_objects.popitem()
        except KeyError:
            messagebox.showerror("Highlight Error", "There is no selection highlighted")
            return

        # we only perform operations on the input grid
        if canvas.winfo_name().__contains__(OUTPUT):
            '''
            display error dialog, reset highlights, reset selections (empties bool dict and reset buttons)
            '''
            messagebox.showerror("Selection Error",
                                 "Boolean selection can only be made from the input grid!"
                                 "\n\nMake a new selection.")
            for r in selected:
                canvas.itemconfig(r, outline='')
            return

        normalized = self.normalize_selection(canvas, selected)
        array = self.selection_to_array(normalized)

        if selection_number == 2:
            '''
            check that current selection size is the same as the first
            if it is not, display error message with correct size and just return so we can try again
            '''
            first_selection = self.bool_op_dict[1]
            if first_selection.shape != array.shape:
                messagebox.showerror("Selection Error",
                                     f"Current selection size {array.shape} is not the same size "
                                     f"as the first selection {first_selection.shape}\n\n"
                                     f"Make a new selection")
                return

        self.bool_op_dict[selection_number] = array

        # set the gui button options based on the section number
        if selection_number == 1:
            self.btn_bool_sel_1.state(['disabled'])  # disable the current button
            self.bool_op_combo.state(['disabled'])
            self.btn_bool_sel_2.state(['!disabled'])  # enable both the second selection button
            self.btn_bool_clear.state(['!disabled'])  # and the clear button (so we can reset)
        elif selection_number == 2:
            self.btn_bool_sel_2.state(['disabled'])  # disable the second selection button (1st btn is disabled)
            self.btn_bool_execute.state(['!disabled'])  # since we made it here enable the Execute button

    def bool_op_execute(self):
        """
        Verify that Sel1, Sel2 have all been selected before executing
        """
        if len(self.bool_op_dict) != 2:
            messagebox.showerror("Boolean Operation Error",
                                 "There must be exactly 2 selections made for the Boolean Operation to work.")
            return

        bool_op_1: np.ndarray = self.bool_op_dict[1]
        bool_op_2: np.ndarray = self.bool_op_dict[2]
        bool_type = self.bool_op_combo.get()

        match bool_type:
            case "AND":
                mask: np.ndarray = np.logical_and(bool_op_1, bool_op_2)
            case "OR":
                mask: np.ndarray = np.logical_or(bool_op_1, bool_op_2)
            case 'NAND':
                mask: np.ndarray = np.logical_not(np.logical_and(bool_op_1, bool_op_2))
            case 'NOR':
                mask: np.ndarray = np.logical_not(np.logical_or(bool_op_1, bool_op_2))
            case 'XOR':
                mask: np.ndarray = np.logical_xor(bool_op_1, bool_op_2)
            case 'XNOR':
                mask: np.ndarray = ~np.bitwise_xor(bool_op_1, bool_op_2)
            case _:
                messagebox.showerror(f"Unknown boolean option: {bool_type}: clearing operation.")
                self.bool_op_clear()
                return

        # Copy the new array to the grid & working set
        output_data = np.zeros_like(mask, dtype=int)
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                output_data[row, col] = self.color_index if mask[row, col] else 0

        self.current_working_set.get_output_data().reset_data(deepcopy(output_data))

        self.output_grid_canvas.delete('all')
        self.output_cells.clear()

        # resize the output grid to match the input grid
        self.entry_output_rows.delete(0, tk.END)
        self.entry_output_rows.insert(0, str(output_data.shape[0]))
        self.entry_output_cols.delete(0, tk.END)
        self.entry_output_cols.insert(0, str(output_data.shape[1]))

        self.resize_grid(OUTPUT)
        self.update_grid(output_data, OUTPUT)
        self.current_working_set.get_output_data().reset_data(deepcopy(output_data))

        self.bool_op_clear()

    def bool_op_clear(self):
        self.bool_op_combo.state(['!disabled'])
        self.btn_bool_sel_1.state(['!disabled'])
        self.btn_bool_sel_2.state(['disabled'])
        self.btn_bool_execute.state(['disabled'])
        self.btn_bool_clear.state(['disabled'])
        self.bool_op_dict.clear()
        self.clear_highlights()

    def change_theme(self, param: str):
        self.ttk_style.theme_use(param)
        background = self.ttk_style.lookup('Main.TFrane', 'background')
        self.color_pallet.configure(bg=background)


def enterButtonPressed(event):
    widget = root.focus_get()
    if isinstance(widget, ttk.Button):
        widget.invoke()


if __name__ == '__main__':
    root = ttkthemes.ThemedTk()
    root.title("AGI Input/Output Problem Creator")
    root.set_theme('black')

    root.bind('<Return>', enterButtonPressed)
    root.resizable(False, False)
    creator = Creator(root)
    creator.default.set('Black')
    root.mainloop()
