import tkinter as tk
from tkinter import messagebox

# File to store inventory data
INVENTORY_FILE = 'inventory.txt'

def add_inventory():
    item_name = item_name_entry.get().strip()
    item_qty = item_qty_entry.get().strip()
    if not item_name or not item_qty:
        messagebox.showwarning("Input error", "Please enter both item name and quantity.")
        return
    try:
        qty = int(item_qty)
        if qty < 0:
            raise ValueError
    except ValueError:
        messagebox.showwarning("Input error", "Quantity must be a non-negative integer.")
        return
    
    # Append new item to file
    with open(INVENTORY_FILE, 'a') as file:
        file.write(f'{item_name},{qty}\n')
    clear_entries()
    messagebox.showinfo("Success", f"Added {item_name} with quantity {qty}.")

def update_inventory():
    item_name = item_name_entry.get().strip()
    item_qty = item_qty_entry.get().strip()
    if not item_name or not item_qty:
        messagebox.showwarning("Input error", "Please enter both item name and quantity.")
        return
    try:
        qty = int(item_qty)
        if qty < 0:
            raise ValueError
    except ValueError:
        messagebox.showwarning("Input error", "Quantity must be a non-negative integer.")
        return
    
    updated = False
    try:
        with open(INVENTORY_FILE, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        messagebox.showwarning("File error", "Inventory file not found.")
        return
    
    with open(INVENTORY_FILE, 'w') as file:
        for line in lines:
            name, old_qty = line.strip().split(',')
            if name.lower() == item_name.lower():
                file.write(f'{name},{qty}\n')
                updated = True
            else:
                file.write(line)
    
    clear_entries()
    if updated:
        messagebox.showinfo("Success", f"Updated {item_name} with quantity {qty}.")
    else:
        messagebox.showinfo("Not found", f"{item_name} not found in inventory.")

def search_inventory():
    search_name = item_name_entry.get().strip()
    if not search_name:
        messagebox.showwarning("Input error", "Please enter item name to search.")
        return
    try:
        with open(INVENTORY_FILE, 'r') as file:
            for line in file:
                name, qty = line.strip().split(',')
                if name.lower() == search_name.lower():
                    result_text.delete(1.0, tk.END)
                    result_text.insert(tk.END, f'{name}: {qty}\n')
                    return
    except FileNotFoundError:
        messagebox.showwarning("File error", "Inventory file not found.")
        return
    
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f'{search_name} not found in inventory.')

def remove_inventory():
    remove_name = item_name_entry.get().strip()
    if not remove_name:
        messagebox.showwarning("Input error", "Please enter item name to remove.")
        return
    
    try:
        with open(INVENTORY_FILE, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        messagebox.showwarning("File error", "Inventory file not found.")
        return
    
    found = False
    with open(INVENTORY_FILE, 'w') as file:
        for line in lines:
            name, qty = line.strip().split(',')
            if name.lower() != remove_name.lower():
                file.write(line)
            else:
                found = True
    
    clear_entries()
    if found:
        messagebox.showinfo("Success", f"Removed {remove_name} from inventory.")
    else:
        messagebox.showinfo("Not found", f"{remove_name} not found in inventory.")

def generate_inventory():
    try:
        with open(INVENTORY_FILE, 'r') as file:
            lines = file.readlines()
            if not lines:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "Inventory is empty.")
                return
            inventory_list = ""
            for line in lines:
                name, qty = line.strip().split(',')
                inventory_list += f"{name} : {qty}\n"
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, inventory_list)
    except FileNotFoundError:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Inventory file not found.")

def clear_entries():
    item_name_entry.delete(0, tk.END)
    item_qty_entry.delete(0, tk.END)

# UI Setup
root = tk.Tk()
root.title("Inventory Management System")

tk.Label(root, text="Item Name:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
item_name_entry = tk.Entry(root)
item_name_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Item Quantity:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
item_qty_entry = tk.Entry(root)
item_qty_entry.grid(row=1, column=1, padx=5, pady=5)

btn_frame = tk.Frame(root)
btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

tk.Button(btn_frame, text="Add Inventory", width=15, command=add_inventory).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Update Inventory", width=15, command=update_inventory).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Search Inventory", width=15, command=search_inventory).grid(row=1, column=0, padx=5, pady=5)
tk.Button(btn_frame, text="Remove Inventory", width=15, command=remove_inventory).grid(row=1, column=1, padx=5, pady=5)
tk.Button(btn_frame, text="Generate Inventory", width=32, command=generate_inventory).grid(row=2, column=0, columnspan=2, pady=5)

result_text = tk.Text(root, height=10, width=40)
result_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()